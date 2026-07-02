"""
Train LoopWM on multiple task types, MSE loss on full obs embedding.

Compares convergence and loop-depth usage across:
  - navigate (simple)
  - collect   (medium)
  - transform (complex)

Key question: does the adaptive loop mechanism use MORE iterations
for harder tasks AFTER training?
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
    sample_loop_count,
)
from textworld_env import (
    TextWorld,
    SimpleTokenizer,
    collect_trajectories,
    build_vocab,
    ACTIONS,
)
from train import LoopWM, TextObsEncoder, ActionEmbedder


# ──────────────────────────────────────────────
#  Per-task adaptive loop analysis
# ──────────────────────────────────────────────

def analyze_loop_depth(
    model: LoopWM,
    val_loaders: dict[str, DataLoader],
    device: torch.device,
    max_test_loops: int = 10,
) -> dict[str, dict]:
    """
    Analyze how many iterations the adaptive mechanism uses for each task type.
    Uses fixed threshold to observe natural loop depth distribution.
    """
    model.eval()
    results = {}

    for task_type, loader in val_loaders.items():
        loop_counts = []
        gate_values = []
        h_norms = []
        cos_sims = []
        mses = []

        with torch.no_grad():
            for batch in loader:
                obs_ids = batch["obs_ids"].to(device)
                obs_mask = batch["obs_mask"].to(device)
                action = batch["action"].to(device)
                next_ids = batch["next_obs_ids"].to(device)
                next_mask = batch["next_obs_mask"].to(device)

                pred_emb, trace = model(
                    obs_ids, obs_mask, action,
                    next_ids, next_mask,
                    num_loops=max_test_loops,
                )
                target = model.encoder(next_ids, next_mask)

                loop_counts.append(len(trace))
                gate_values.append([t['gate_mean'] for t in trace])

                for t in trace:
                    h_norms.append(t['h_norm'])

                cos_sims.append(F.cosine_similarity(pred_emb, target, dim=-1).mean().item())
                mses.append(F.mse_loss(pred_emb, target).item())

        avg_loops = sum(loop_counts) / len(loop_counts)
        max_loops = max(loop_counts)
        min_loops = min(loop_counts)

        # If early_exit_threshold were set to 0.9, how many would exit early?
        # Simulate: count samples where any gate > 0.9 at any step
        early_exit_90 = sum(1 for g in gate_values if any(v > 0.90 for v in g))
        early_exit_80 = sum(1 for g in gate_values if any(v > 0.80 for v in g))

        results[task_type] = {
            "avg_loops": avg_loops,
            "min_loops": min_loops,
            "max_loops": max_loops,
            "samples": len(loop_counts),
            "avg_cos_sim": sum(cos_sims) / len(cos_sims),
            "avg_mse": sum(mses) / len(mses),
            "pct_gate_above_90": early_exit_90 / len(loop_counts) * 100,
            "pct_gate_above_80": early_exit_80 / len(loop_counts) * 100,
        }

    return results


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    TASK_TYPES = ["navigate", "collect", "transform"]

    # ── Collect data ──
    print("\nCollecting multi-task data...")
    all_data = {}
    for task in TASK_TYPES:
        all_data[task] = collect_trajectories(
            num_episodes=100, task_type=task,
            seed=42 + TASK_TYPES.index(task) * 1000,
        )
        print(f"  {task}: {len(all_data[task]['observations'])} transitions")

    # Train/val split per task
    train_data = {}
    val_data = {}
    for task in TASK_TYPES:
        d = all_data[task]
        split = int(len(d["observations"]) * 0.8)
        train_data[task] = {k: v[:split] for k, v in d.items()}
        val_data[task] = {k: v[split:] for k, v in d.items()}

    # Mixed dataset
    mixed_train = {"observations": [], "actions": [], "rewards": [],
                   "dones": [], "episode_ids": []}
    mixed_val = {"observations": [], "actions": [], "rewards": [],
                 "dones": [], "episode_ids": []}
    for task in TASK_TYPES:
        for k in mixed_train:
            mixed_train[k].extend(train_data[task][k])
            mixed_val[k].extend(val_data[task][k])

    print(f"\n  Mixed train: {len(mixed_train['observations'])} transitions")
    print(f"  Mixed val:   {len(mixed_val['observations'])} transitions")

    # ── Build tokenizer ──
    from textworld_env import SimpleTokenizer
    tokenizer = SimpleTokenizer()
    all_texts = list(mixed_train['observations']) + [a.lower() for a in ACTIONS]
    tokenizer.fit(all_texts)
    tokenizer.freeze()
    vocab_size = tokenizer.vocab_size
    print(f"  Vocab size: {vocab_size}")

    # ── Datasets ──
    from train import SequenceDataset, collate_fn

    train_ds = SequenceDataset(mixed_train, tokenizer)
    val_ds = SequenceDataset(mixed_val, tokenizer)

    train_loader = DataLoader(
        train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    # Per-task val loaders
    per_task_loaders = {}
    for task in TASK_TYPES:
        ds = SequenceDataset(val_data[task], tokenizer)
        per_task_loaders[task] = DataLoader(
            ds, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=0
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
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)

    # ── Training ──
    print(f"{'Epoch':>5} {'Loss':>8} {'ValMSE':>8} {'CosSim':>6} "
          f"{'Gate':>6} {'Loops':>6} {'Time':>6}")
    print("-" * 55)

    for epoch in range(80):
        t0 = time.time()

        # --- Train ---
        model.train()
        total_loss = 0.0
        n = 0
        gate_means = []
        loop_counts = []
        norm_stable = True

        # Epoch 0-20: K=3, mu=4; 20+: K=5, mu=7
        if epoch < 20:
            K, mu = 3, 4.0
        else:
            K, mu = 5, 7.0

        for batch in train_loader:
            obs_ids = batch["obs_ids"].to(device)
            obs_mask = batch["obs_mask"].to(device)
            action = batch["action"].to(device)
            next_ids = batch["next_obs_ids"].to(device)
            next_mask = batch["next_obs_mask"].to(device)

            optimizer.zero_grad()

            num_loops = sample_loop_count(mu=mu, max_t=K if epoch < 20 else 10)
            pred_emb, trace = model(
                obs_ids, obs_mask, action, next_ids, next_mask,
                num_loops=num_loops,
            )

            with torch.no_grad():
                target = model.encoder(next_ids, next_mask)
            loss = F.mse_loss(pred_emb, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            b = obs_ids.size(0)
            total_loss += loss.item() * b
            n += b

            if trace:
                gate_means.append(trace[-1]['gate_mean'])
                loop_counts.append(len(trace))
            for t in trace:
                if t['h_norm'] > 1e4:
                    norm_stable = False

        # --- Validate ---
        model.eval()
        val_mse = 0.0
        cos_sim_sum = 0.0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                obs_ids = batch["obs_ids"].to(device)
                obs_mask = batch["obs_mask"].to(device)
                action = batch["action"].to(device)
                next_ids = batch["next_obs_ids"].to(device)
                next_mask = batch["next_obs_mask"].to(device)

                pred_emb, _ = model(obs_ids, obs_mask, action, next_ids, next_mask)
                target = model.encoder(next_ids, next_mask)

                val_mse += F.mse_loss(pred_emb, target, reduction='sum').item()
                cos_sim_sum += F.cosine_similarity(pred_emb, target, dim=-1).sum().item()
                total += pred_emb.size(0)

        avg_loss = total_loss / max(n, 1)
        avg_mse = val_mse / max(total, 1)
        avg_cos = cos_sim_sum / max(total, 1)
        avg_gate = sum(gate_means) / max(len(gate_means), 1)
        avg_loops = sum(loop_counts) / max(len(loop_counts), 1)
        norm_str = "OK" if norm_stable else "⚠️"
        elapsed = time.time() - t0

        print(f"{epoch:5d} {avg_loss:8.6f} {avg_mse:8.6f} {avg_cos:6.4f} "
              f"{avg_gate:6.4f} {avg_loops:6.2f} {elapsed:6.1f}s")

        if math.isnan(avg_loss):
            print(f"\n❌ DIVERGED at epoch {epoch}")
            break

        # Early stopping
        if epoch >= 15 and avg_cos > 0.96:
            print(f"\n✅ Converged at epoch {epoch} (CosSim={avg_cos:.4f})")
            break

    # ── Post-training: per-task adaptive depth analysis ──
    print("\n" + "=" * 60)
    print("  Per-Task Adaptive Depth Analysis")
    print("=" * 60)

    # Analysis 1: with normal threshold (0.9)
    results = analyze_loop_depth(model, per_task_loaders, device, max_test_loops=10)

    print(f"\n{'Task':<15} {'MSE':>8} {'CosSim':>8} {'AvgLoops':>10} "
          f"{'Min':>5} {'Max':>5} {'Gate>90%':>8}")
    print("-" * 60)

    for task in TASK_TYPES:
        r = results[task]
        print(f"{task:<15} {r['avg_mse']:8.6f} {r['avg_cos_sim']:8.4f} "
              f"{r['avg_loops']:10.2f} {r['min_loops']:5d} {r['max_loops']:5d} "
              f"{r['pct_gate_above_90']:7.1f}%")

    # Analysis 2: gate threshold sweep
    print(f"\nGate threshold sweep (what % would exit early at each threshold):")
    print(f"{'Task':<15} ", end="")
    for th in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
        print(f"{th:>6}", end="")
    print()

    for task in TASK_TYPES:
        print(f"{task:<15} ", end="")
        loader = per_task_loaders[task]
        model.eval()
        with torch.no_grad():
            gate_thresholds = {th: 0 for th in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]}
            total_samples = 0
            for batch in loader:
                obs_ids = batch["obs_ids"].to(device)
                obs_mask = batch["obs_mask"].to(device)
                action = batch["action"].to(device)
                next_ids = batch["next_obs_ids"].to(device)
                next_mask = batch["next_obs_mask"].to(device)

                _, trace = model(obs_ids, obs_mask, action, next_ids, next_mask,
                                 num_loops=10)
                gate_vals = [t['gate_mean'] for t in trace]
                total_samples += 1

                for th in gate_thresholds:
                    if any(g > th for g in gate_vals):
                        gate_thresholds[th] += 1

            for th in sorted(gate_thresholds.keys()):
                pct = gate_thresholds[th] / max(total_samples, 1) * 100
                print(f"{pct:6.1f}", end="")
        print()

    # ── Save ──
    torch.save(model.state_dict(), "loopwm_multitask_mse.pt")
    print(f"\nModel saved to loopwm_multitask_mse.pt")
    print(f"Total params: {total_params:,}")


if __name__ == "__main__":
    main()
