""" Fitness cache for reusing evaluation results across similar architectures.

Similarity is measured by cosine distance between softmax-normalized alpha vectors,
making it applicable to both discrete (PMF-derived alpha) and mixedop mode architectures.

On a cache hit, the cached fitness/params/inference-time are reused directly and no
training happens at all (no weight loading, no fine-tuning) - this trades search-time
accuracy for speed, on the assumption that architectures with near-identical alpha
vectors would train to near-identical fitness anyway.
"""

import json
import os
import numpy as np
from typing import Optional


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    return float(1.0 - np.dot(a_norm, b_norm))


def alpha_to_signature(alpha_vec: np.ndarray) -> str:
    """Convert a flat softmax-normalized alpha vector to a string signature."""
    probs = _softmax(alpha_vec.flatten())
    return ";".join(f"{v:.3f}" for v in probs.tolist())


def signature_to_vec(sig: str) -> np.ndarray:
    return np.array([float(v) for v in sig.split(";")])


class WeightBank:
    """JSON-indexed cache of (fitness, params, inference_time, weights_path) results,
    keyed by architecture alpha signature.

    Workers only READ the index (snapshot before generation). The main process writes
    the index after all workers join, so no locking is needed.
    """

    def __init__(self, bank_dir: str, cosine_threshold: float = 0.05):
        self.bank_dir = bank_dir
        self.cosine_threshold = cosine_threshold
        self.index_path = os.path.join(bank_dir, "index.json")
        self._index: dict = {}
        os.makedirs(bank_dir, exist_ok=True)
        self._load_index()

    def _load_index(self):
        if os.path.exists(self.index_path):
            with open(self.index_path, "r") as f:
                self._index = json.load(f)

    def _save_index(self):
        with open(self.index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    def find_cached_result(self, alpha_vec: np.ndarray) -> Optional[dict]:
        """Return the cached result of the closest architecture, or None if no match.

        Args:
            alpha_vec: flat (or multi-dim) alpha logit array for the candidate architecture.

        Returns:
            {"weights_path", "fitness", "params_m", "inference_us"} of the closest
            cached architecture within cosine_threshold, or None.
        """
        if not self._index:
            return None

        query = _softmax(alpha_vec.flatten())
        best_dist = float("inf")
        best_entry = None

        for sig, entry in self._index.items():
            if not os.path.exists(entry["weights_path"]):
                continue
            ref = signature_to_vec(sig)
            d = _cosine_distance(query, ref)
            if d < best_dist and d <= self.cosine_threshold:
                best_dist = d
                best_entry = entry

        return best_entry

    def register(self, alpha_vec: np.ndarray, weights_path: str, fitness: float,
                 params_m: float, inference_us: float):
        """Register an evaluated architecture's result in the bank (main process only)."""
        sig = alpha_to_signature(alpha_vec)
        self._index[sig] = {
            "weights_path": weights_path,
            "fitness": fitness,
            "params_m": params_m,
            "inference_us": inference_us,
        }
        self._save_index()

    def snapshot(self) -> "WeightBank":
        """Return a read-only copy of the bank (for subprocess use)."""
        snap = WeightBank.__new__(WeightBank)
        snap.bank_dir = self.bank_dir
        snap.cosine_threshold = self.cosine_threshold
        snap.index_path = self.index_path
        snap._index = dict(self._index)
        return snap
