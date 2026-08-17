"""Small tensor operations shared by MeCaSNet model variants."""

from __future__ import annotations

import torch


def scatter_logsumexp(
    source: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
) -> torch.Tensor:
    """Compute a numerically stable log-sum-exp over groups on dimension 0.

    ``index[i]`` assigns ``source[i]`` to one of ``dim_size`` groups. The
    function supports scalar edge scores ``(E,)`` and vector scores ``(E, D)``
    without requiring the optional ``torch-scatter`` package.
    """
    if source.ndim < 1:
        raise ValueError("source must have at least one dimension")
    if index.ndim != 1 or index.shape[0] != source.shape[0]:
        raise ValueError("index must be one-dimensional and match source.shape[0]")
    if dim_size < 0:
        raise ValueError("dim_size must be non-negative")

    output_shape = (dim_size, *source.shape[1:])
    if source.shape[0] == 0:
        return torch.full(output_shape, -torch.inf, dtype=source.dtype, device=source.device)

    expanded_index = index.reshape(-1, *([1] * (source.ndim - 1))).expand_as(source)
    maxima = torch.full(output_shape, -torch.inf, dtype=source.dtype, device=source.device)
    maxima.scatter_reduce_(0, expanded_index, source, reduce="amax", include_self=True)

    gathered_maxima = maxima.index_select(0, index)
    shifted = torch.exp(source - gathered_maxima)
    totals = torch.zeros(output_shape, dtype=source.dtype, device=source.device)
    totals.index_add_(0, index, shifted)

    result = maxima + torch.log(totals.clamp_min(torch.finfo(source.dtype).tiny))
    return torch.where(torch.isfinite(maxima), result, maxima)

