"""
🔥 火行·焱 — Iterative retrieval for AgMem + LoopWM.

Extends AgMem's 4-tier 火行 recall (温→烟→燃→炎) with an
iterative refinement stage using RecurrentDynamicsKernel.

Usage:
    from path_b.iterative_recall import IterativeRecall, demo
    ir = IterativeRecall(db)
    results = ir.recall("python programming", top_k=3)
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# Try torch + loopwm kernel
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    import sys
    _kernel_path = str(Path(__file__).resolve().parent.parent / "kernel")
    if _kernel_path not in sys.path:
        sys.path.insert(0, _kernel_path)
    from recurrent import RecurrentDynamicsKernel
    TORCH_OK = True
except ImportError as e:
    TORCH_OK = False
    logger.warning("IterativeRecall: torch/kernel not available — %s", e)


# ──────────────────────────────────────────────
#  QueryEncoder: text ↔ embedding
# ──────────────────────────────────────────────

class QueryEncoder:
    """
    Simple bag-of-words encoder: build vocab from texts,
    encode as bag-of-embeddings averaged, decode by finding
    nearest vocab words.
    """

    def __init__(self, d_model: int = 128):
        self.d_model = d_model
        self.word_to_id: dict[str, int] = {"<pad>": 0}
        self.id_to_word: dict[int, str] = {0: "<pad>"}
        self.embed: nn.Embedding | None = None

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().replace(",", " ,").replace(".", " .").split()

    def fit(self, texts: list[str]):
        seen = set()
        for t in texts:
            for w in self._tokenize(t):
                if w not in seen and w not in self.word_to_id:
                    idx = len(self.word_to_id)
                    self.word_to_id[w] = idx
                    self.id_to_word[idx] = w
                    seen.add(w)

    def build_embed(self):
        if self.embed is None and len(self.word_to_id) > 1:
            self.embed = nn.Embedding(len(self.word_to_id), self.d_model)

    def encode(self, query: str) -> torch.Tensor:
        """[1, d_model] — mean-pooled bag of word embeddings."""
        self.build_embed()
        ids = []
        for w in self._tokenize(query):
            idx = self.word_to_id.get(w)
            if idx is not None:
                ids.append(idx)
        if not ids or self.embed is None:
            return torch.zeros(1, self.d_model)
        emb = self.embed(torch.tensor([ids], dtype=torch.long))
        return emb.mean(dim=1)  # [1, d_model]

    def decode(self, emb: torch.Tensor, top_k: int = 5) -> list[str]:
        """Nearest vocab words to embedding."""
        if self.embed is None:
            return []
        V = self.embed.weight.shape[0]
        sims = F.cosine_similarity(emb, self.embed.weight.unsqueeze(0), dim=-1)
        top_ids = sims.topk(min(top_k, V)).indices.squeeze(0).tolist()
        if isinstance(top_ids, int):
            top_ids = [top_ids]
        return [self.id_to_word.get(i, "?") for i in top_ids if i != 0]


# ──────────────────────────────────────────────
#  IterativeRecall
# ──────────────────────────────────────────────

class IterativeRecall:
    """
    Iterative retrieval: refine query → retrieve → repeat.

    Uses RecurrentDynamicsKernel to iteratively refine the query
    embedding through retrieval-fuse cycles before final retrieval.
    """

    def __init__(
        self,
        db,
        d_model: int = 128,
        max_iters: int = 5,
        exit_threshold: float = 0.85,
    ):
        self.db = db
        self.max_iters = max_iters
        self.exit_threshold = exit_threshold

        if not TORCH_OK:
            self.encoder = None
            self.kernel = None
            return

        # Build vocab from existing memory
        all_nodes = db.get_all_nodes() if hasattr(db, 'get_all_nodes') else []
        all_texts = [n.content for n in all_nodes]

        self.encoder = QueryEncoder(d_model=d_model)
        self.encoder.fit(all_texts)
        self.encoder.build_embed()

        self.kernel = RecurrentDynamicsKernel(
            d_model=d_model, max_loops=max_iters,
            early_exit_threshold=exit_threshold,
        )
        self.kernel.eval()

    def recall(self, query: str, top_k: int = 5, **kwargs) -> list[dict]:
        """Iterative retrieval → final results."""
        if not TORCH_OK or self.kernel is None:
            return self._std(query, top_k, **kwargs)

        # Iterative refinement
        q_vec = self.encoder.encode(query)  # [1, d_model]
        h = q_vec.clone()
        trace_log = []

        for t in range(self.max_iters):
            # Decode current embedding → words → simulated retrieval signal
            words = self.encoder.decode(h, top_k=5)
            refined = " ".join(words)

            # Score: word overlap with original query
            q_words = set(self.encoder._tokenize(query))
            r_words = set(words)
            overlap = len(q_words & r_words) / max(len(q_words | r_words), 1) if (q_words or r_words) else 0
            trace_log.append({"step": t, "refined": refined, "overlap": overlap})

            # Kernel step: h, e_k=current_embed, u_k=original_query
            e_k = h.clone()
            u_k = q_vec.clone()
            h_new, ktrace = self.kernel(h, e_k, u_k)
            h = h_new

            if ktrace and ktrace[-1]['gate_mean'] > self.exit_threshold:
                logger.debug("early exit at step %d (gate=%.4f)", t, ktrace[-1]['gate_mean'])
                break

        # Final retrieval with refined query
        final_words = self.encoder.decode(h, top_k=10)
        final_q = " ".join(final_words)
        return self._std(final_q, top_k, **kwargs)

    def _std(self, query: str, top_k: int = 5, **kwargs) -> list[dict]:
        """Standard retrieval (fallback or final)."""
        try:
            from agmem.retrieval import memory_recall
            return memory_recall(self.db, query, top_k=top_k, **kwargs)
        except ImportError:
            return []


# ──────────────────────────────────────────────
#  Demo
# ──────────────────────────────────────────────

def demo():
    """Populate AgMem, compare single-shot vs iterative recall."""
    from agmem.core import AgMemDB
    from agmem.retrieval import retrieve, RecallTier

    db = AgMemDB(":memory:")

    # Two clusters: programming pythons, snake pythons
    prog = [
        "Python is a programming language for backend development",
        "Django and FastAPI are Python web frameworks",
        "Python packages are managed with pip",
    ]
    snakes = [
        "Ball pythons are popular pet snakes",
        "Burmese pythons can grow over 20 feet long",
        "Pythons are non-venomous constrictors",
    ]

    nodes = {}
    for text in prog + snakes:
        node = db.create_node(
            node_id=str(uuid.uuid4()),
            content=text, node_type="fact", strength=0.9,
        )
        nodes[text] = node

    # Links within clusters
    for cluster in [prog, snakes]:
        for i in range(1, len(cluster)):
            db.create_link(nodes[cluster[i]].id, nodes[cluster[i-1]].id,
                          "related_to", 0.5)

    # ── Compare ──
    ambiguous = "python programming"

    single = retrieve(db, ambiguous, top_k=3, tier=RecallTier.BURN)
    print("  Single-shot (燃 tier):")
    for p in single:
        txt = p.nodes[0].content[:60] if p.nodes else "?"
        print(f"    · {txt}")

    ir = IterativeRecall(db, d_model=64)
    iterative = ir.recall(ambiguous, top_k=3)
    print("\n  Iterative:")
    for r in iterative:
        print(f"    · {r.get('content', '?')[:60]}")

    # Score: how many results are from the programming cluster?
    def cluster_count(results, cluster_texts):
        """Handle both ScoredPath (from retrieve) and dict (from memory_recall)."""
        def get_content(r):
            if hasattr(r, 'get'):
                return str(r.get('content', ''))
            if hasattr(r, 'nodes') and r.nodes:
                return str(r.nodes[0].content)
            return str(r)
        return sum(1 for r in results
                   for ref in cluster_texts
                   if ref[:20] in get_content(r))

    s_prog = cluster_count(single, prog)
    i_prog = cluster_count(iterative, prog)
    print(f"\n  Programming cluster hits: single={s_prog}/3  iterative={i_prog}/3")

    return db


if __name__ == "__main__":
    demo()
