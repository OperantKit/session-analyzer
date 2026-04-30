"""Operant-variability analysis (Page & Neuringer, 1985).

The *U-value* is the Shannon entropy of an observed distribution over
response sequences, normalized to the maximum entropy achievable under
a uniform distribution across all possible sequences. It lies in
``[0, 1]``:

- ``U = 1``: responses are uniformly distributed over all possible
  sequences (maximum variability).
- ``U = 0``: all responses fall into a single sequence.

Given a fixed alphabet of operanda of size ``A`` and a fixed sequence
length ``L``, there are ``N = A^L`` possible sequences, and
``U = H / log(N)`` where ``H`` is the Shannon entropy of the observed
sequence distribution.

Reference
---------
Page, S., & Neuringer, A. (1985). Variability is an operant.
    *Journal of Experimental Psychology: Animal Behavior Processes*,
    *11*(3), 429-452. https://doi.org/10.1037/0097-7403.11.3.429
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Sequence


@dataclass(frozen=True)
class VariabilityMetrics:
    """Operant-variability summary.

    Attributes
    ----------
    u_value : float
        Normalized entropy ``H / log(N)`` in ``[0, 1]``.
    entropy_bits : float
        Shannon entropy of the observed sequence distribution, in bits.
    n_sequences : int
        Total number of sequences observed (sample size).
    n_unique : int
        Number of unique sequences observed.
    n_possible : int
        Total possible sequences given the alphabet size and length
        (``A**L``).
    """

    u_value: float
    entropy_bits: float
    n_sequences: int
    n_unique: int
    n_possible: int


def compute_u_value(
    sequences: Sequence[Hashable],
    *,
    alphabet_size: int,
    sequence_length: int,
) -> VariabilityMetrics:
    """Compute the Page & Neuringer (1985) U-value.

    Parameters
    ----------
    sequences : sequence of hashable
        Observed response sequences (strings, tuples of ints, etc.);
        any hashable representation works as long as each element
        names exactly one sequence.
    alphabet_size : int
        Number of distinct operanda (``A``, strictly > 1).
    sequence_length : int
        Length of each observed sequence (``L``, strictly positive).
        All observed sequences should share this length; sequences of
        any other length raise ``ValueError``.

    Returns
    -------
    VariabilityMetrics

    Raises
    ------
    ValueError
        If ``sequences`` is empty, ``alphabet_size < 2``,
        ``sequence_length < 1``, or any sequence has an unexpected
        length.
    """
    if len(sequences) == 0:
        raise ValueError("sequences must be non-empty")
    if alphabet_size < 2:
        raise ValueError(f"alphabet_size must be >= 2, got {alphabet_size}")
    if sequence_length < 1:
        raise ValueError(f"sequence_length must be >= 1, got {sequence_length}")

    for s in sequences:
        try:
            s_len = len(s)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError(
                "sequences must support len(); got an element without length"
            ) from exc
        if s_len != sequence_length:
            raise ValueError(
                f"all sequences must have length {sequence_length}; "
                f"found length {s_len}"
            )

    counts = Counter(sequences)
    n = sum(counts.values())
    entropy = 0.0
    for c in counts.values():
        p = c / n
        entropy -= p * math.log2(p)

    n_possible = alphabet_size**sequence_length
    max_entropy = math.log2(n_possible)
    u = entropy / max_entropy if max_entropy > 0.0 else 0.0

    return VariabilityMetrics(
        u_value=float(u),
        entropy_bits=float(entropy),
        n_sequences=int(n),
        n_unique=len(counts),
        n_possible=int(n_possible),
    )
