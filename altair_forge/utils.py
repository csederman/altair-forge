"""General plotting utilities for Altair Forge."""

from __future__ import annotations

import math
import typing as t


def get_domain(values: t.Iterable[float], step: float = 1) -> t.Tuple[float, float]:
    """Computes a sensible domain for a set of values.

    Parameters
    ----------
    values : t.Iterable[float]
        The values to compute the domain for.
    step : float, optional
        The step size to use when computing the domain, by default 1.

    Returns
    -------
    t.Tuple[float, float]
        A tuple of (min, max) values for the domain.
    """
    inv = 1 / step
    min_ = math.floor(min(values) * inv) / inv
    max_ = math.ceil(max(values) * inv) / inv
    return min_, max_
