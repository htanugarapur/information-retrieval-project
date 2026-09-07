"""Retrieval metrics and the statistics the ablation results are reported with.

Rule enforced by this module's return types: nothing here returns a bare mean.
`Distribution` carries the spread and the confidence interval alongside it, so a
caller physically cannot report a centre without one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


# ------------------------------------------------------------- ranking


def dcg_at_k(relevances: Sequence[float], k: int) -> float:
    total = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        total += (2.0 ** rel - 1.0) / math.log2(i + 1.0)
    return total


def ndcg_at_k(
    ranked_ids: Sequence[str], relevance: Mapping[str, float], k: int = 10
) -> float:
    """NDCG@k against a graded relevance map. Returns 0.0 when nothing is relevant."""
    gains = [float(relevance.get(paper_id, 0.0)) for paper_id in ranked_ids[:k]]
    ideal = sorted((float(v) for v in relevance.values()), reverse=True)[:k]
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg <= 0.0:
        return 0.0
    return dcg_at_k(gains, k) / ideal_dcg


def reciprocal_rank(ranked_ids: Sequence[str], relevant: Iterable[str]) -> float:
    relevant_set = set(relevant)
    for i, paper_id in enumerate(ranked_ids, start=1):
        if paper_id in relevant_set:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(
    rankings: Sequence[Sequence[str]], relevant_sets: Sequence[Iterable[str]]
) -> float:
    if not rankings:
        return 0.0
    scores = [reciprocal_rank(r, rel) for r, rel in zip(rankings, relevant_sets)]
    return float(np.mean(scores))


# --------------------------------------------------------- distributions


@dataclass(frozen=True)
class Distribution:
    """A sample summarised so it can never be quoted without its spread."""

    n: int
    mean: float
    median: float
    std: float
    ci_low: float
    ci_high: float
    confidence: float
    minimum: float
    maximum: float

    def format(self, digits: int = 3) -> str:
        return (
            f"{self.mean:.{digits}f} "
            f"[{self.ci_low:.{digits}f}, {self.ci_high:.{digits}f}] "
            f"(n={self.n})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "confidence": self.confidence,
            "min": self.minimum,
            "max": self.maximum,
        }


EMPTY_DISTRIBUTION = Distribution(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.95, 0.0, 0.0)


def bootstrap_distribution(
    values: Sequence[float],
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Distribution:
    """Percentile bootstrap CI of the mean.

    Seeded so a reported interval is reproducible: an unseeded CI that shifts
    between runs cannot be checked by a reviewer.
    """
    array = np.asarray([float(v) for v in values], dtype=np.float64)
    if array.size == 0:
        return EMPTY_DISTRIBUTION
    if array.size == 1:
        value = float(array[0])
        return Distribution(1, value, value, 0.0, value, value, confidence, value, value)

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(samples, array.size))
    means = array[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0

    return Distribution(
        n=int(array.size),
        mean=float(array.mean()),
        median=float(np.median(array)),
        std=float(array.std(ddof=1)),
        ci_low=float(np.quantile(means, alpha)),
        ci_high=float(np.quantile(means, 1.0 - alpha)),
        confidence=confidence,
        minimum=float(array.min()),
        maximum=float(array.max()),
    )


# ------------------------------------------------------------- testing


@dataclass(frozen=True)
class PairedTest:
    """Result of comparing an explanation's effect against its control."""

    name: str
    n: int
    statistic: float
    p_value: float
    mean_difference: float
    effect_size: float
    significant: bool
    alpha: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "test": self.name,
            "n": self.n,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "mean_difference": self.mean_difference,
            "effect_size": self.effect_size,
            "significant": self.significant,
            "alpha": self.alpha,
        }


def paired_permutation_test(
    treatment: Sequence[float],
    control: Sequence[float],
    samples: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PairedTest:
    """Two-sided paired permutation (sign-flip) test on the differences.

    Chosen over a paired t-test because rank displacements are bounded, discrete
    and badly non-normal -- most candidates do not move at all, so the
    distribution has a large spike at zero that violates normality outright.
    """
    a = np.asarray([float(v) for v in treatment], dtype=np.float64)
    b = np.asarray([float(v) for v in control], dtype=np.float64)
    if a.size != b.size:
        raise ValueError("paired test needs equal-length samples")
    if a.size == 0:
        return PairedTest("paired_permutation", 0, 0.0, 1.0, 0.0, 0.0, False, alpha)

    differences = a - b
    observed = float(differences.mean())

    if np.allclose(differences, 0.0):
        return PairedTest("paired_permutation", int(a.size), 0.0, 1.0, 0.0, 0.0, False, alpha)

    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(samples, differences.size))
    null_means = (signs * differences).mean(axis=1)
    # +1 in numerator and denominator: an exact Monte Carlo p-value can never be
    # reported as 0, which would overstate certainty from a finite resample.
    p_value = float((np.sum(np.abs(null_means) >= abs(observed)) + 1) / (samples + 1))

    spread = float(differences.std(ddof=1)) if differences.size > 1 else 0.0
    effect = observed / spread if spread > 0 else 0.0

    return PairedTest(
        name="paired_permutation",
        n=int(a.size),
        statistic=observed,
        p_value=p_value,
        mean_difference=observed,
        effect_size=effect,
        significant=p_value < alpha,
        alpha=alpha,
    )


def monte_carlo_p_value(observed: float, null_samples: Sequence[float]) -> float:
    """One-sided p: how often the control matched or beat the real ablation.

    This is the per-candidate faithfulness test. With `n` control trials the
    smallest achievable p is 1/(n+1), so `ablate.random_control_trials` must be
    large enough for that floor to sit under the chosen alpha -- the CLI checks
    this at startup rather than letting every case silently fail to reach
    significance.
    """
    null = np.asarray([float(v) for v in null_samples], dtype=np.float64)
    if null.size == 0:
        return 1.0
    return float((np.sum(null >= observed) + 1) / (null.size + 1))


def min_achievable_p_value(trials: int) -> float:
    return 1.0 / (trials + 1)
