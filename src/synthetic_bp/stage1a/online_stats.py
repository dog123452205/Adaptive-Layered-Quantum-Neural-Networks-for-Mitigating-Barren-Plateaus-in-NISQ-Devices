"""
Online statistics for Stage 1A.

Welford aggregation is used to compute Var_seed[dE/dθ_k]
without storing all raw gradient components.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass
class OnlineGradientStats:
    """
    Online statistics for one fixed parameter gradient.

    Attributes:
        count:
            Number of observed seeds.
        mean:
            Running mean of gradient value.
        m2:
            Running sum of squared deviations.
        abs_sum:
            Running sum of absolute gradient values.
        near_zero_count:
            Number of near-zero gradient observations.
    """
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    abs_sum: float = 0.0
    near_zero_count: int = 0

    def update(self, value: float, near_zero_threshold: float) -> None:
        """
        Update online statistics with one gradient value.

        Args:
            value:
                Gradient value for a fixed parameter under one random seed.
            near_zero_threshold:
                Threshold used for near-zero counting.
        """
        self.count += 1

        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean

        self.m2 += delta * delta2
        self.abs_sum += abs(value)

        if abs(value) < near_zero_threshold:
            self.near_zero_count += 1

    @property
    def variance(self) -> float:
        """
        Unbiased sample variance.

        Returns:
            Variance across seeds.
        """
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def mean_abs(self) -> float:
        """
        Mean absolute gradient.

        Returns:
            Mean absolute gradient.
        """
        if self.count == 0:
            return 0.0
        return self.abs_sum / self.count

    @property
    def near_zero_ratio(self) -> float:
        """
        Fraction of near-zero observations.

        Returns:
            Near-zero ratio.
        """
        if self.count == 0:
            return 0.0
        return self.near_zero_count / self.count