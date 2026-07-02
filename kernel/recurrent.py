"""
RecurrentDynamicsKernel — the core of Looped World Models.

Implements the parameter-shared looped transformer block with spectral stability
constraint, adaptive early exit, and Prelude/Recurrent/Coda architecture.

Paper: arxiv:2606.18208v1 — Looped World Models
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
#  Spectral Stability (论文 Eq.6–7, 3 行核心)
# ──────────────────────────────────────────────

def spectral_stabilize(h: torch.Tensor, a_log: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
    """
    Apply spectral stability constraint: A_bar = exp(-dt * exp(a_log)).

    Because -dt * exp(a_log) is always negative, exp() of that is always in (0, 1).
    This guarantees the state transition matrix has eigenvalues strictly < 1,
    preventing explosion regardless of iteration count.

    Args:
        h: Latent state [B, d_model]
        a_log: Learnable log-parameters [d_model]
        dt: Time step (controls decay speed)
    Returns:
        h_stabilized: [B, d_model]
    """
    a_bar = torch.exp(-dt * torch.exp(a_log))  # 自动 ∈ (0, 1)
    return a_bar.unsqueeze(0) * h  # [B, d_model]


def compute_a_bar(a_log: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
    """
    Compute the spectral decay vector for inspection/debugging.

    Returns: [d_model] with all values in (0, 1).
    """
    return torch.exp(-dt * torch.exp(a_log))


# ──────────────────────────────────────────────
#  Adaptive Early Exit
# ──────────────────────────────────────────────

def early_exit_gate(h: torch.Tensor, gate_proj: nn.Linear,
                    threshold: float = 0.85) -> tuple[torch.Tensor, float]:
    """
    Compute adaptive early exit decision.

    Args:
        h: Current state [B, d_model]
        gate_proj: Learned projection -> scalar gate
        threshold: Exit when gate.mean() > threshold
    Returns:
        (gate, gate_mean): gate tensor [B, 1] and its batch-mean scalar
    """
    gate = torch.sigmoid(gate_proj(h))  # [B, 1]
    gate_mean = gate.mean().item()
    return gate, gate_mean


# ──────────────────────────────────────────────
#  RecurrentDynamicsKernel
# ──────────────────────────────────────────────

class RecurrentDynamicsKernel(nn.Module):
    """
    Parameter-shared looped dynamics core — the heart of LoopWM.

    Architecture:
        Prelude ℙ:  Input fusion (h_prev, e_k, u_k) → conditioning signal e
        Recurrent ℝ: Parameter-shared transformer block, looped T times
                     with spectral stability + residual connection + early exit
        Coda ℂ:      Final refinement on terminal state

    Usage:
        kernel = RecurrentDynamicsKernel(d_model=128)
        h_k, trace = kernel(h_prev, e_k, u_k)
        # trace contains per-step diagnostics: gate_mean, h_norm
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        dt: float = 1.0,
        max_loops: int = 10,
        early_exit_threshold: float = 0.9,
    ):
        super().__init__()
        self.d_model = d_model
        self.dt = dt
        self.max_loops = max_loops
        self.early_exit_threshold = early_exit_threshold

        # — Spectral stability (论文 Eq.6–7) —
        self.a_log = nn.Parameter(torch.randn(d_model) * 0.1)

        # — Prelude 𝓟: Input fusion —
        # 3x because we concat [h_prev; e_k; u_k]
        self.prelude = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
        )

        # — Recurrent block 𝓡: Parameter-shared transformer —
        # Single encoder layer, applied T times with shared params
        self.recurrent_block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )

        # — Input injection 𝓑̄ (learnable) —
        self.B_bar = nn.Linear(d_model, d_model, bias=False)

        # — Adaptive early exit gate —
        self.gate_proj = nn.Linear(d_model, 1)

        # — Coda ℂ: Final refinement —
        self.coda = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )

    def _spectral_stabilize(self, h: torch.Tensor) -> torch.Tensor:
        """论文 Eq.3: A_bar = exp(-dt * exp(a_log)), 自动 ∈ (0,1)"""
        return spectral_stabilize(h, self.a_log, self.dt)

    def forward(
        self,
        h_prev: torch.Tensor,
        e_k: torch.Tensor,
        u_k: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, list[dict]]:
        """
        Args:
            h_prev: Previous latent state [B, d_model]
            e_k: Observation embedding / conditioning signal [B, d_model]
            u_k: Action embedding [B, d_model] or None (zero-filled)
        Returns:
            h_k: Refined latent state [B, d_model]
            trace: Per-step diagnostics
        """
        B = h_prev.size(0)
        device = h_prev.device

        if u_k is None:
            u_k = torch.zeros_like(e_k)

        # — Prelude: Condition signal e —
        x = torch.cat([h_prev, e_k, u_k], dim=-1)  # [B, 3*d_model]
        e = self.prelude(x)  # [B, d_model]

        # — Recurrent block: T iterations with shared params —
        h = h_prev  # h^(0)
        trace: list[dict] = []

        for t in range(self.max_loops):
            # 1) Spectral stability: h ← Ā·h
            h_stable = self._spectral_stabilize(h)  # Ā·h^(t)

            # 2) Input injection: + B̄·e
            h_injected = h_stable + self.B_bar(e)

            # 3) Recurrent transformer: ℝ̄(h, e)
            # TransformerEncoderLayer expects [B, seq_len, D]; we use seq_len=1
            h_transformed = self.recurrent_block(h_injected.unsqueeze(1))

            # 4) Residual connection: h^(t+1) = Ā·h^(t) + B̄·e + ℝ̄(h^(t), e)
            h = h_injected + h_transformed.squeeze(1)

            # 5) Early exit gate
            gate, gate_mean = early_exit_gate(h, self.gate_proj,
                                               self.early_exit_threshold)

            trace.append({
                'step': t,
                'gate_mean': gate_mean,
                'gate_std': gate.std().item(),
                'h_norm': h.norm(dim=-1).mean().item(),
                'h_stable_norm': h_stable.norm(dim=-1).mean().item(),
                'a_bar_min': compute_a_bar(self.a_log, self.dt).min().item(),
                'a_bar_max': compute_a_bar(self.a_log, self.dt).max().item(),
            })

            if gate_mean > self.early_exit_threshold and t >= 1:
                break

        # — Coda: Final refinement —
        h_k = self.coda(h.unsqueeze(1)).squeeze(1)  # [B, d_model]

        return h_k, trace


# ──────────────────────────────────────────────
#  Poisson Schedule (论文 §3.3 Variable-Depth Training)
# ──────────────────────────────────────────────

def sample_loop_count(mu: float = 5.0, max_t: int = 10) -> int:
    """
    Sample loop count from Poisson distribution, per-sequence independent.

    Paper: "T ~ Poisson(μ_rec), sampled independently per sequence to reduce
    training variance compared to batch-level sampling."
    """
    t = max(1, torch.poisson(torch.tensor(mu)).int().item())
    return min(t, max_t)


# ──────────────────────────────────────────────
#  Observation Header (for A: ScienceWorld)
# ──────────────────────────────────────────────

class ObservationHead(nn.Module):
    """
    Predicts the next observation's latent embedding (MSE reconstruction).

    Instead of predicting a single token, we predict the mean-pooled embedding
    of the entire next observation — forcing the model to capture full content.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, h_k: torch.Tensor) -> torch.Tensor:
        """Returns predicted embedding [B, d_model]"""
        return self.proj(h_k)


# ──────────────────────────────────────────────
#  Curriculum Schedule (论文 §3.3)
# ──────────────────────────────────────────────

class CurriculumSchedule:
    """
    Two-phase curriculum: K-step unfolding horizon.

    Phase A (epochs 0-N_A): K=2, μ=5  — learn short horizon
    Phase B (epochs N_A+):  K=5, μ=10 — learn long horizon
    """

    def __init__(self, phase_a_epochs: int = 30):
        self.phase_a_epochs = phase_a_epochs

    def get_params(self, epoch: int) -> tuple[int, float]:
        if epoch < self.phase_a_epochs:
            return 2, 5.0  # K, μ
        else:
            return 5, 10.0
