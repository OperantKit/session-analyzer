"""N-gram entropy analyses for behavioural sequences.

Beyond the U-value (which measures the total entropy of fixed-length
sequence frequencies), conditional and order-N entropies expose the
*dependence structure* of a response stream:

- :func:`compute_block_entropy` — Shannon entropy of length-``n``
  blocks, ``H_n = -sum p(s_n) log2 p(s_n)``.
- :func:`compute_conditional_entropy` — entropy of the next symbol
  given the preceding ``order``-length context, ``h_n = H_{n+1} - H_n``.
- :func:`estimate_markov_order` — smallest ``n`` for which
  ``h_n - h_{n+1}`` falls below ``tolerance`` (a heuristic Markov
  order estimate).

References
----------
Hopson, J. W., Burt, D. B., & Hopson, J. M. (2002). RGCalc: A program
    for calculating randomization indices. *Behavior Research
    Methods, Instruments, & Computers*, *34*(1), 127-128.
    https://doi.org/10.3758/BF03195461

Cover, T. M., & Thomas, J. A. (2006). *Elements of information
    theory* (2nd ed.). Wiley.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Sequence


@dataclass(frozen=True)
class BlockEntropy:
    """Block entropy of length-``n`` substrings.

    Attributes
    ----------
    order : int
        Block length ``n``.
    entropy_bits : float
        ``H_n`` in bits.
    n_blocks : int
    n_unique_blocks : int
    """

    order: int
    entropy_bits: float
    n_blocks: int
    n_unique_blocks: int


@dataclass(frozen=True)
class ConditionalEntropy:
    """Conditional entropy ``h_n = H_{n+1} - H_n``.

    Interpreted as the bits of uncertainty about the next symbol
    given the preceding ``order``-length context.

    Attributes
    ----------
    order : int
    h_n_bits : float
    block_n_entropy : float
        ``H_n``.
    block_np1_entropy : float
        ``H_{n+1}``.
    """

    order: int
    h_n_bits: float
    block_n_entropy: float
    block_np1_entropy: float


def _block_counts(
    symbols: Sequence[Hashable], block_len: int
) -> tuple[Counter, int]:
    if block_len < 1:
        raise ValueError(f"block_len must be >= 1, got {block_len}")
    if len(symbols) < block_len:
        raise ValueError(
            f"need at least {block_len} symbols for block length {block_len}, "
            f"got {len(symbols)}"
        )
    counts: Counter = Counter()
    for i in range(len(symbols) - block_len + 1):
        block = tuple(symbols[i : i + block_len])
        counts[block] += 1
    total = sum(counts.values())
    return counts, total


def _entropy(counts: Counter, total: int) -> float:
    h = 0.0
    for c in counts.values():
        p = c / total
        h -= p * math.log2(p)
    return h


def compute_block_entropy(
    symbols: Sequence[Hashable], *, order: int
) -> BlockEntropy:
    """Compute the block entropy ``H_order``."""
    counts, total = _block_counts(symbols, order)
    return BlockEntropy(
        order=int(order),
        entropy_bits=float(_entropy(counts, total)),
        n_blocks=int(total),
        n_unique_blocks=len(counts),
    )


def compute_conditional_entropy(
    symbols: Sequence[Hashable], *, order: int
) -> ConditionalEntropy:
    """Compute the conditional entropy ``h_order = H_{order+1} - H_order``.

    For ``order = 0``, this returns the marginal symbol entropy.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    if order == 0:
        marginal = compute_block_entropy(symbols, order=1)
        return ConditionalEntropy(
            order=0,
            h_n_bits=marginal.entropy_bits,
            block_n_entropy=0.0,
            block_np1_entropy=marginal.entropy_bits,
        )
    h_n = compute_block_entropy(symbols, order=order)
    h_np1 = compute_block_entropy(symbols, order=order + 1)
    return ConditionalEntropy(
        order=int(order),
        h_n_bits=float(h_np1.entropy_bits - h_n.entropy_bits),
        block_n_entropy=float(h_n.entropy_bits),
        block_np1_entropy=float(h_np1.entropy_bits),
    )


@dataclass(frozen=True)
class MarkovOrderEstimate:
    """Heuristic Markov-order estimate by entropy-rate plateau detection.

    Attributes
    ----------
    estimated_order : int
        The smallest ``order`` for which ``h_{order} - h_{order+1}``
        falls below ``tolerance``. Reported as ``max_order`` if no
        plateau is reached within the search window.
    h_rates : tuple of float
        Conditional entropies ``h_0, h_1, ..., h_{max_order}``.
    tolerance : float
    converged : bool
    """

    estimated_order: int
    h_rates: tuple[float, ...]
    tolerance: float
    converged: bool


def estimate_markov_order(
    symbols: Sequence[Hashable],
    *,
    max_order: int = 4,
    tolerance: float = 0.05,
) -> MarkovOrderEstimate:
    """Estimate Markov order from the conditional-entropy plateau.

    Parameters
    ----------
    symbols : sequence of hashable
        Symbol stream.
    max_order : int
        Upper search bound (inclusive). Default 4.
    tolerance : float
        Minimum ``h_n - h_{n+1}`` decrement that still counts as
        non-plateaued. Default 0.05 bits.
    """
    if max_order < 1:
        raise ValueError(f"max_order must be >= 1, got {max_order}")
    if tolerance <= 0.0:
        raise ValueError(f"tolerance must be positive, got {tolerance}")

    h_rates: list[float] = []
    for n in range(0, max_order + 1):
        h_rates.append(compute_conditional_entropy(symbols, order=n).h_n_bits)

    estimated = max_order
    converged = False
    for n in range(len(h_rates) - 1):
        if h_rates[n] - h_rates[n + 1] < tolerance:
            estimated = n
            converged = True
            break
    return MarkovOrderEstimate(
        estimated_order=int(estimated),
        h_rates=tuple(h_rates),
        tolerance=float(tolerance),
        converged=converged,
    )
