""" Weight bank for reusing trained model weights across similar architectures.

Similarity is measured by cosine distance between softmax-normalized alpha vectors,
making it applicable to both discrete (PMF-derived alpha) and mixedop mode architectures.
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple


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
    """JSON-indexed cache of trained model weights, keyed by architecture alpha signature.

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

    def find_close_match(self, alpha_vec: np.ndarray) -> Optional[str]:
        """Return the weights path of the closest architecture, or None if no match.

        Args:
            alpha_vec: flat (or multi-dim) alpha logit array for the candidate architecture.

        Returns:
            Path string to the closest best_model.pth, or None.
        """
        if not self._index:
            return None

        query = _softmax(alpha_vec.flatten())
        best_dist = float("inf")
        best_path = None

        for sig, path in self._index.items():
            if not os.path.exists(path):
                continue
            ref = signature_to_vec(sig)
            d = _cosine_distance(query, ref)
            if d < best_dist and d <= self.cosine_threshold:
                best_dist = d
                best_path = path

        return best_path

    def save_weights(self, alpha_vec: np.ndarray, weights_path: str):
        """Register a trained model in the bank (main process only)."""
        sig = alpha_to_signature(alpha_vec)
        self._index[sig] = weights_path
        self._save_index()

    def load_weights_into_model(self, model: nn.Module, weights_path: str,
                                 device: str) -> bool:
        """Load saved weights into model with strict=False (partial loading OK)."""
        try:
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            return True
        except Exception:
            return False

    def snapshot(self) -> "WeightBank":
        """Return a read-only copy of the bank (for subprocess use)."""
        snap = WeightBank.__new__(WeightBank)
        snap.bank_dir = self.bank_dir
        snap.cosine_threshold = self.cosine_threshold
        snap.index_path = self.index_path
        snap._index = dict(self._index)
        return snap
