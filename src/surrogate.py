""" Surrogate-assisted candidate screening (direction D1 of
docs/QNAS_SEARCH_ENGINE_IMPROVEMENT_PLAN_V2.md).

A bootstrap ensemble of ridge regressions over the one-hot (node position x op) encoding of
an architecture predicts its fitness from the evaluations already made in the current
progressive stage. Offline, on the cache.json of the experiment_cifar_pqnas runs, this
ranked the second half of a stage's evaluations from the first half with Spearman 0.69-0.96.

Candidates are then picked by Boltzmann sampling without replacement (Gumbel top-k on
U / tau, U = z-score of mu + beta * sigma), with a fraction of uniformly random picks as an
off-policy safeguard. Only NumPy; every random draw goes through the caller's own
np.random.Generator, never the global numpy RNG the default search path consumes.
"""

import numpy as np


def one_hot(net_pop, num_functions):
    """ [n, num_nodes] int gene indices -> [n, sum(num_functions)] one-hot features.

    Args:
        net_pop: int ndarray of gene indices (a negative gene encodes no op and gets no 1).
        num_functions: list with the number of ops available at each node.
    """
    net_pop = np.asarray(net_pop)
    offsets = np.concatenate(([0], np.cumsum(num_functions)[:-1])).astype(np.int64)
    features = np.zeros((net_pop.shape[0], int(np.sum(num_functions))), dtype=np.float64)
    rows, nodes = np.nonzero(net_pop >= 0)
    features[rows, offsets[nodes] + net_pop[rows, nodes]] = 1.0
    return features


def zscore(values):
    values = np.asarray(values, dtype=np.float64)
    std = values.std()
    return (values - values.mean()) / std if std > 1e-12 else np.zeros_like(values)


def gumbel_topk(scores, tau, n, rng):
    """ n indices sampled without replacement from p proportional to exp(scores / tau). """
    n = min(n, scores.shape[0])
    noisy = scores / max(tau, 1e-8) + rng.gumbel(size=scores.shape[0])
    return np.argsort(-noisy, kind='stable')[:n]


class RidgeEnsembleSurrogate:
    """ Bootstrap ensemble of ridge regressions on one-hot architecture features. """

    def __init__(self, num_models=5, ridge_lambda=3.0, rng=None):
        self.num_models = num_models
        self.ridge_lambda = ridge_lambda
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.coefs = []
        self.feature_mean = None
        self.target_mean = 0.0

    @property
    def fitted(self):
        return bool(self.coefs)

    def fit(self, features, targets):
        """ Fit every member on a bootstrap resample. Non-finite targets (diverged
            individuals) are dropped. Returns self. """
        features = np.asarray(features, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        finite = np.isfinite(targets)
        features, targets = features[finite], targets[finite]
        self.coefs = []
        if targets.shape[0] < 2:
            return self
        self.feature_mean = features.mean(axis=0)
        self.target_mean = float(targets.mean())
        centered = features - self.feature_mean
        y = targets - self.target_mean
        eye = self.ridge_lambda * np.eye(features.shape[1])
        for _ in range(self.num_models):
            idx = self.rng.integers(0, targets.shape[0], targets.shape[0])
            a = centered[idx]
            self.coefs.append(np.linalg.solve(a.T @ a + eye, a.T @ y[idx]))
        return self

    def predict(self, features):
        """ (mean, std) of the ensemble's predictions. """
        centered = np.asarray(features, dtype=np.float64) - self.feature_mean
        preds = np.stack([centered @ w for w in self.coefs]) + self.target_mean
        return preds.mean(axis=0), preds.std(axis=0)

    def op_effects(self, num_functions):
        """ Mean ensemble coefficient per (node, op): the estimated marginal effect of using
            that op at that node, as one array per node (used by prune_criterion
            'surrogate'). """
        coef = np.mean(self.coefs, axis=0)
        bounds = np.concatenate(([0], np.cumsum(num_functions)))
        return [coef[bounds[i]:bounds[i + 1]].copy() for i in range(len(num_functions))]


def screen(candidates, num_functions, train_net, train_fit, n_select, tau, beta,
           random_frac, rng, num_models=5, ridge_lambda=3.0):
    """ Pick *n_select* rows of *candidates* to actually evaluate.

    Args:
        candidates: [n_candidates, num_nodes] int gene indices.
        num_functions: ops per node (the current stage's menu).
        train_net, train_fit: evaluations of the current stage the surrogate learns from.
        n_select: how many candidates to keep.
        tau: Boltzmann temperature on the z-scored utility.
        beta: weight of the ensemble's spread in the utility (exploration bonus).
        random_frac: fraction of n_select filled uniformly at random from the remaining
            candidates, regardless of the surrogate.
        rng: np.random.Generator used for every random draw.

    Returns:
        (selected_indices, predicted_mean) - predicted_mean is aligned with *candidates*.
    """
    n_candidates = candidates.shape[0]
    n_select = min(n_select, n_candidates)
    model = RidgeEnsembleSurrogate(num_models=num_models, ridge_lambda=ridge_lambda, rng=rng)
    model.fit(one_hot(train_net, num_functions), train_fit)
    if not model.fitted:
        chosen = rng.choice(n_candidates, size=n_select, replace=False)
        return np.sort(chosen), np.full(n_candidates, np.nan)

    mu, sigma = model.predict(one_hot(candidates, num_functions))
    utility = zscore(mu + beta * sigma)
    n_random = int(round(random_frac * n_select))
    n_guided = n_select - n_random
    guided = gumbel_topk(utility, tau, n_guided, rng) if n_guided > 0 else np.array([], int)
    rest = np.setdiff1d(np.arange(n_candidates), guided)
    random_pick = rng.choice(rest, size=min(n_random, rest.shape[0]), replace=False) \
        if n_random > 0 and rest.shape[0] > 0 else np.array([], int)
    return np.concatenate((guided, random_pick)).astype(np.int64), mu
