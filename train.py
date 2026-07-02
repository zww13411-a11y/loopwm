"""
Train LoopWM on TextWorld synthetic data.

Pipeline:
  1. Collect trajectories (obs_text, action, reward, done)
  2. Build tokenizer
  3. obs_text → token embeddings → mean pool → e_k (encoder ℰ_φ)
  4. action → action embedding → u_k (action embedder 𝒜_ψ)
  5. h_k = RecurrentDynamicsKernel(h_prev, e_k, u_k)
  6. h_k → predict next obs tokens (observation head 𝒟_ξ)
  7. Loss = cross-entropy(next_tokens, predicted_logits)
  8. Backprop, repeat
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).parent))

from kernel.recurrent import (
    RecurrentDynamicsKernel,
    ObservationHead,
    CurriculumSchedule,
    sample_loop_count,
)
from textworld_env import (
    TextWorld,
    SimpleTokenizer,
    collect_trajectories,
    build_vocab,
    ACTIONS,
)


# ──────────────────────────────────────────────
#  Text Encoder (ℰ_φ)
# ──────────────────────────────────────────────

class TextObsEncoder(nn.Module):
    """
    Encodes text observations into latent embeddings.

    Architecture: Token Embed → 2-layer Transformer Encoder → Mean Pool
    """

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int = 64):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model * 4,
            dropout=0.1, activation='gelu', batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor | None = None
                ) -> torch.Tensor:
        """
        Args:
            token_ids: [B, seq_len]
            mask: [B, seq_len] — 1 for valid, 0 for pad
        Returns:
            [B, d_model] — mean-pooled embedding
        """
        B, S = token_ids.shape
        x = self.token_embed(token_ids)  # [B, S, d_model]
        x = x + self.pos_embed[:, :S, :]
        x = self.encoder(x)
        if mask is not None:
            mask = mask.unsqueeze(-1).float()  # [B, S, 1]
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        return x  # [B, d_model]


# ──────────────────────────────────────────────
#  Action Embedder (𝒜_ψ)
# ──────────────────────────────────────────────

class ActionEmbedder(nn.Module):
    """
    Embeds discrete actions into the same latent space.

    Minimal: Embedding lookup (action_id → d_model).
    """

    def __init__(self, num_actions: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(num_actions, d_model)

    def forward(self, action_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action_ids: [B]
        Returns:
            [B, d_model]
        """
        return self.embed(action_ids)


# ──────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────

class SequenceDataset(Dataset):
    """
    Dataset of (obs_tokens, action, next_obs_tokens, reward, done) sequences.

    Each item is a single transition from the trajectory.
    """

    def __init__(self, data: dict, tokenizer: SimpleTokenizer, max_seq_len: int = 64):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.obs_tokens = []
        self.next_obs_tokens = []
        self.actions = []
        self.rewards = []
        self.dones = []

        for i in range(len(data["observations"])):
            obs_tok = self._tokenize(data["observations"][i])
            next_obs = self._tokenize(data["observations"][
                i + 1 if i + 1 < len(data["observations"]) else i
            ])
            self.obs_tokens.append(obs_tok)
            self.next_obs_tokens.append(next_obs)
            self.actions.append(data["actions"][i])
            self.rewards.append(data["rewards"][i])
            self.dones.append(data["dones"][i])

    def _tokenize(self, text: str) -> list[int]:
        tokens = self.tokenizer.encode(text)
        return tokens[:self.max_seq_len]

    def __len__(self):
        return len(self.obs_tokens)

    def __getitem__(self, idx):
        return {
            "obs_ids": torch.tensor(self.obs_tokens[idx], dtype=torch.long),
            "obs_mask": torch.ones(len(self.obs_tokens[idx])),
            "action": torch.tensor(self.actions[idx], dtype=torch.long),
            "next_obs_ids": torch.tensor(self.next_obs_tokens[idx], dtype=torch.long),
            "next_obs_mask": torch.ones(len(self.next_obs_tokens[idx])),
            "reward": torch.tensor(self.rewards[idx], dtype=torch.float),
            "done": torch.tensor(self.dones[idx], dtype=torch.float),
        }


def collate_fn(batch, pad_token_id: int = 0):
    """Collate variable-length sequences into padded batches."""
    max_obs_len = max(item["obs_ids"].size(0) for item in batch)
    max_next_len = max(item["next_obs_ids"].size(0) for item in batch)

    obs_ids = torch.full((len(batch), max_obs_len), pad_token_id, dtype=torch.long)
    obs_mask = torch.zeros(len(batch), max_obs_len)
    next_obs_ids = torch.full((len(batch), max_next_len), pad_token_id, dtype=torch.long)
    next_obs_mask = torch.zeros(len(batch), max_next_len)

    for i, item in enumerate(batch):
        l = item["obs_ids"].size(0)
        obs_ids[i, :l] = item["obs_ids"]
        obs_mask[i, :l] = 1

        l2 = item["next_obs_ids"].size(0)
        next_obs_ids[i, :l2] = item["next_obs_ids"]
        next_obs_mask[i, :l2] = 1

    return {
        "obs_ids": obs_ids,
        "obs_mask": obs_mask,
        "action": torch.stack([item["action"] for item in batch]),
        "next_obs_ids": next_obs_ids,
        "next_obs_mask": next_obs_mask,
        "reward": torch.stack([item["reward"] for item in batch]),
        "done": torch.stack([item["done"] for item in batch]),
    }


# ──────────────────────────────────────────────
#  Full LoopWM Model
# ──────────────────────────────────────────────

class LoopWM(nn.Module):
    """
    Full Looped World Model assembled from components.

    ℰ_φ → 𝒜_ψ → ℒ_θ → 𝒟_ξ
    """

    def __init__(
        self,
        vocab_size: int,
        num_actions: int,
        d_model: int = 128,
        max_seq_len: int = 64,
        max_loops: int = 10,
        dt: float = 1.0,
    ):
        super().__init__()
        self.encoder = TextObsEncoder(vocab_size, d_model, max_seq_len)
        self.action_embedder = ActionEmbedder(num_actions, d_model)
        self.dynamics_kernel = RecurrentDynamicsKernel(
            d_model=d_model,
            max_loops=max_loops,
            dt=dt,
            early_exit_threshold=0.90,
        )
        self.obs_head = ObservationHead(d_model)

    def forward(
        self,
        obs_ids: torch.Tensor,
        obs_mask: torch.Tensor,
        action: torch.Tensor,
        next_obs_ids: torch.Tensor | None = None,
        next_obs_mask: torch.Tensor | None = None,
        num_loops: int | None = None,
    ) -> tuple[torch.Tensor, list[dict]]:
        """
        Args:
            obs_ids: [B, S] — current observation tokens
            obs_mask: [B, S] — padding mask
            action: [B] — action indices
            next_obs_ids: [B, S'] — next observation tokens (for target embedding)
            next_obs_mask: [B, S'] — padding mask for next obs
            num_loops: override loop count (for curriculum training)
        Returns:
            pred_embedding: [B, d_model] — predicted next-obs embedding
            trace: per-step diagnostics
        """
        B = obs_ids.size(0)
        device = obs_ids.device

        # Encode current obs
        e_k = self.encoder(obs_ids, obs_mask)  # [B, d_model]
        u_k = self.action_embedder(action)       # [B, d_model]

        # Previous state: zero-init
        h_prev = torch.zeros(B, self.dynamics_kernel.d_model, device=device)

        # Dynamics kernel
        if num_loops is not None:
            old_max = self.dynamics_kernel.max_loops
            self.dynamics_kernel.max_loops = num_loops
            h_k, trace = self.dynamics_kernel(h_prev, e_k, u_k)
            self.dynamics_kernel.max_loops = old_max
        else:
            h_k, trace = self.dynamics_kernel(h_prev, e_k, u_k)

        # Decode: predict next obs embedding
        pred_embedding = self.obs_head(h_k)  # [B, d_model]

        return pred_embedding, trace


# ──────────────────────────────────────────────
#  Training Loop
# ──────────────────────────────────────────────

def train_epoch(
    model: LoopWM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    curriculum: CurriculumSchedule,
    epoch: int,
    device: torch.device,
) -> dict:
    """Train one epoch and return metrics."""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    gate_means = []
    loop_counts = []
    norm_stable = True

    K, mu = curriculum.get_params(epoch)

    for batch in loader:
        obs_ids = batch["obs_ids"].to(device)
        obs_mask = batch["obs_mask"].to(device)
        action = batch["action"].to(device)
        next_ids = batch["next_obs_ids"].to(device)
        next_mask = batch["next_obs_mask"].to(device)

        B, S = obs_ids.shape
        optimizer.zero_grad()

        # Sample variable loop count per sequence (Poisson)
        num_loops = sample_loop_count(mu=mu, max_t=10)
        pred_embedding, trace = model(
            obs_ids, obs_mask, action,
            next_obs_ids=next_ids, next_obs_mask=next_mask,
            num_loops=num_loops,
        )

        # Loss: MSE between predicted embedding and encoded next-obs embedding
        # Encode next observation as target
        with torch.no_grad():
            target_embedding = model.encoder(next_ids, next_mask)  # [B, d_model]
        loss = F.mse_loss(pred_embedding, target_embedding)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * B
        total_tokens += B

        # Metrics
        if trace:
            gate_means.append(trace[-1]['gate_mean'])
            loop_counts.append(len(trace))

        # Check norm stability
        for t in trace:
            if t['h_norm'] > 1e4:
                norm_stable = False

    avg_loss = total_loss / max(total_tokens, 1)
    avg_gate = sum(gate_means) / max(len(gate_means), 1)
    avg_loops = sum(loop_counts) / max(len(loop_counts), 1)

    return {
        "loss": avg_loss,
        "gate": avg_gate,
        "loops": avg_loops,
        "norm_stable": norm_stable,
        "K": K,
        "mu": mu,
    }


def validate(
    model: LoopWM,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Run validation with MSE embedding loss + cosine similarity."""
    model.eval()
    total_mse = 0.0
    total_tokens = 0
    total_cos_sim = 0.0

    with torch.no_grad():
        for batch in loader:
            obs_ids = batch["obs_ids"].to(device)
            obs_mask = batch["obs_mask"].to(device)
            action = batch["action"].to(device)
            next_ids = batch["next_obs_ids"].to(device)
            next_mask = batch["next_obs_mask"].to(device)

            pred_embedding, trace = model(
                obs_ids, obs_mask, action,
                next_obs_ids=next_ids, next_obs_mask=next_mask,
            )
            target_embedding = model.encoder(next_ids, next_mask)

            mse = F.mse_loss(pred_embedding, target_embedding, reduction='sum')
            total_mse += mse.item()

            # Cosine similarity
            cos = F.cosine_similarity(pred_embedding, target_embedding, dim=-1)
            total_cos_sim += cos.sum().item()
            total_tokens += pred_embedding.size(0)

    avg_mse = total_mse / max(total_tokens, 1)
    avg_cos = total_cos_sim / max(total_tokens, 1)

    return {"mse": avg_mse, "cos_sim": avg_cos}


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Collect data ──
    print("Collecting trajectories from TextWorld...")
    train_data = collect_trajectories(num_episodes=200, task_type="navigate", seed=42)
    val_data = collect_trajectories(num_episodes=50, task_type="navigate", seed=999)
    print(f"  Train: {len(train_data['observations'])} transitions")
    print(f"  Val:   {len(val_data['observations'])} transitions")

    # ── Build tokenizer ──
    tokenizer = build_vocab(train_data)
    vocab_size = tokenizer.vocab_size
    print(f"  Vocab size: {vocab_size}")

    # ── Datasets ──
    train_ds = SequenceDataset(train_data, tokenizer)
    val_ds = SequenceDataset(val_data, tokenizer)

    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    # ── Model ──
    model = LoopWM(
        vocab_size=vocab_size,
        num_actions=len(ACTIONS),
        d_model=128,
        max_loops=10,
        dt=1.0,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    curriculum = CurriculumSchedule(phase_a_epochs=30)

    # ── Training ──
    print("\nStarting training...\n")
    print(f"{'Epoch':>5} {'K':>2} {'μ':>3} {'Loss':>8} {'MSE':>8} {'CosSim':>6} "
          f"{'Gate':>6} {'Loops':>6} {'Norm?':>6} {'Time':>6}")
    print("-" * 70)

    for epoch in range(60):
        t0 = time.time()

        train_metrics = train_epoch(model, train_loader, optimizer, curriculum,
                                     epoch, device)
        val_metrics = validate(model, val_loader, device)

        norm_str = "OK" if train_metrics["norm_stable"] else "⚠️"
        elapsed = time.time() - t0

        print(f"{epoch:5d} {int(train_metrics['K']):2d} {train_metrics['mu']:3.0f} "
              f"{train_metrics['loss']:8.4f} {val_metrics['mse']:8.4f} "
              f"{val_metrics['cos_sim']:6.3f} {train_metrics['gate']:6.4f} "
              f"{train_metrics['loops']:6.2f} {norm_str:>6} {elapsed:6.1f}s")

        # Check for divergence
        if train_metrics["loss"] > 100 or math.isnan(train_metrics["loss"]):
            print(f"\n❌ LOSS DIVERGED at epoch {epoch}! Training stopped.")
            break

        # Early stopping if converged
        if epoch >= 10 and val_metrics["cos_sim"] > 0.95:
            print(f"\n✅ Converged! CosSim > 0.95 at epoch {epoch}")
            break

    # ── Final report ──
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)

    # Final validation
    final_val = validate(model, val_loader, device)
    print(f"\nFinal validation:")
    print(f"  Train Loss: {train_metrics['loss']:.4f}")
    print(f"  Val MSE:    {final_val['mse']:.6f}")
    print(f"  CosSim:     {final_val['cos_sim']:.4f} ({final_val['cos_sim']*100:.1f}%)")
    print(f"  Params:     {total_params:,}")
    print(f"  d_model:    128")

    # Check gate distribution
    print(f"\nGate distribution (final epoch):")
    print(f"  Mean:       {train_metrics['gate']:.4f}")
    print(f"  Avg loops:  {train_metrics['loops']:.2f}")

    # Save model
    torch.save(model.state_dict(), "loopwm_trained.pt")
    print("\nModel saved to loopwm_trained.pt")


if __name__ == "__main__":
    main()
