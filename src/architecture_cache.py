""" Fitness cache for classical individuals: keyed by the exact architecture (net_list)
found, so that if the same architecture reappears (e.g. via elitism carrying it
forward, or the population converging), its fitness is reused instead of retraining.

No weights are stored or reused - only the (architecture -> fitness/params/inference
time) mapping, plus a hit_count tracking how many times each architecture has been
reused from the cache. On a cache hit, the cached values are reused directly and no
training happens at all.

A single cache.json lives in the experiment folder and is shared across the whole run
(all progressive stages included) - the key already encodes the full architecture
(net_list), so individuals from different stages (different node counts/op menus)
naturally produce different keys and never collide.

Updated live, per individual: each worker process registers an architecture's result
as soon as that individual finishes training, instead of batching updates until the
whole generation completes - so a duplicate architecture that appears later in the
same generation (a different worker process) can already hit the cache. Since workers
are separate OS processes evaluating individuals concurrently, every read and write
goes through this same on-disk cache.json under an flock-based lock, so no in-memory
snapshot can go stale and no concurrent write can corrupt the file or lose an update.
"""

import fcntl
import json
import os
from typing import List, Optional


def net_list_signature(net_list: List[str]) -> str:
    """Convert a decoded network (list of op names) to a stable string key."""
    return "|".join(net_list)


class ArchitectureCache:
    """File-backed, multi-process-safe cache of (fitness, params, inference_time,
    hit_count) results, keyed by the exact architecture (net_list) that produced them.

    Every read and write re-opens and locks *cache_path* itself (flock, POSIX advisory
    lock) rather than relying on an in-memory copy, so concurrent worker processes
    always see each other's latest updates and never corrupt the file.
    """

    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        # Ensure the file exists without writing any content to it: "a" mode
        # creates-if-missing and is a no-op if it already exists, and since we never
        # call .write() here, this can't race with (or corrupt) another process's
        # locked read-modify-write in register()/find_cached_result(). An empty file
        # is handled as {} below.
        open(cache_path, "a").close()

    def _locked_read_modify_write(self, mutate):
        """Open cache_path once, hold an exclusive lock for the whole read-modify-
        write cycle, and flush/fsync before releasing it - the shared primitive behind
        both find_cached_result() (increments hit_count on a hit) and register().

        Args:
            mutate: callable(index: dict) -> Any, mutates *index* in place and
                returns whatever the caller wants back.

        Returns:
            Whatever *mutate* returned.
        """
        with open(self.cache_path, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.seek(0)
                content = f.read()
                index = json.loads(content) if content else {}
                result = mutate(index)
                f.seek(0)
                f.truncate()
                json.dump(index, f, indent=2)
                # Flush the write to the OS before releasing the lock - otherwise the
                # lock could be released (letting another process in) while these
                # bytes are still sitting in Python's userspace buffer, unflushed.
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        return result

    def find_cached_result(self, net_list: List[str]) -> Optional[dict]:
        """Return the cached result for this exact architecture, or None if unseen.
        On a hit, atomically increments that entry's hit_count in the cache.

        Args:
            net_list: decoded network (list of op names) for the candidate individual.

        Returns:
            {"fitness", "params_m", "inference_us", "hit_count"} if this exact
            architecture has already been evaluated, or None.
        """
        key = net_list_signature(net_list)

        def mutate(index):
            entry = index.get(key)
            if entry is not None:
                entry["hit_count"] = entry.get("hit_count", 0) + 1
                index[key] = entry
            return entry

        return self._locked_read_modify_write(mutate)

    def register(self, net_list: List[str], fitness: float, params_m: float,
                 inference_us: float):
        """Register an evaluated architecture's result in the cache, merging with
        whatever other worker processes have already written (locked read-modify-write
        of the shared cache.json - safe to call concurrently from multiple processes).
        Preserves the existing hit_count if this architecture was already registered.
        """
        key = net_list_signature(net_list)

        def mutate(index):
            existing = index.get(key, {})
            index[key] = {
                "fitness": fitness,
                "params_m": params_m,
                "inference_us": inference_us,
                "hit_count": existing.get("hit_count", 0),
            }

        self._locked_read_modify_write(mutate)
