# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""The split combine on its own: random partials and partial LSEs merged against an FP32 torch
reference, with NaN in every partial slot past a row's valid split count so a masking defect
shows up as a non-finite output. Covers the split counts, head counts, partial and output
dtypes, the temperature LSE and the output scale the forward kernels can hand it."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch

from fmha_sm100.prefill_q8kv4.interface import _sparse_stack

from .conftest import run_timed

HEAD_DIM = 128
Q_HEADS_PER_KV = 16


@dataclass(frozen=True)
class CombineCase:
    name: str
    total_q: int
    num_kv_heads: int
    topk: int
    partial_dtype: torch.dtype = torch.bfloat16
    out_dtype: torch.dtype = torch.bfloat16
    split_counts: bool = True
    lse_out: bool = True
    temperature: bool = False
    output_scale: bool = False


CASES = (
    CombineCase("h64_topk16", 333, 4, 16),
    # 16 heads: the flat rows are not a multiple of the tile, so the last tile is partial
    CombineCase("h16_topk16_partial_tile", 171, 1, 16),
    CombineCase("h32_topk4", 100, 2, 4),
    CombineCase("h64_topk8_f16_partials", 257, 4, 8, partial_dtype=torch.float16),
    CombineCase("h64_topk32_temperature_scale", 129, 4, 32, temperature=True, output_scale=True),
    CombineCase("h64_topk16_every_split", 65, 4, 16, split_counts=False, lse_out=False),
    CombineCase("one_query_f16_out", 1, 1, 16, out_dtype=torch.float16),
)


def _real_to_fake_col(device: torch.device) -> torch.Tensor:
    """Fake (stored) column of every real column: inside each 32-column group the forward
    epilogue stores a 4x4 transpose of the 32-bit pairs."""
    col = torch.arange(HEAD_DIM, device=device)
    group_base, col32 = col // 32 * 32, col % 32
    return group_base + (col32 % 8) // 2 * 8 + col32 // 8 * 2 + col32 % 2


def _reference(partial, lse, valid, output_scale):
    """FP32 merge of the valid splits: softmax weights over the partial LSEs, final LSE."""
    lse = torch.where(valid, lse, torch.full_like(lse, -math.inf))
    row_max = lse.amax(dim=0)
    safe_max = torch.where(torch.isinf(row_max), torch.zeros_like(row_max), row_max)
    weight = torch.exp(lse - safe_max)
    total = weight.sum(dim=0)
    inv_total = torch.where(total > 0, 1.0 / total, torch.zeros_like(total))
    partial = torch.where(valid[..., None], partial.float(), torch.zeros_like(partial, dtype=torch.float32))
    out = ((weight * inv_total)[..., None] * partial).sum(dim=0) * output_scale
    final_lse = torch.where(total > 0, row_max + torch.log(total), torch.full_like(row_max, -math.inf))
    return out, final_lse


def _nan_fill(tensor: torch.Tensor, mask: torch.Tensor) -> None:
    """Write a quiet NaN into every element selected by ``mask`` (broadcast over trailing dims)."""
    nan = torch.tensor(math.nan, dtype=tensor.dtype, device=tensor.device)
    tensor.masked_fill_(mask.reshape(mask.shape + (1,) * (tensor.ndim - mask.ndim)), nan)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_combine_merges_the_valid_splits(case: CombineCase, device: torch.device):
    _, combine = _sparse_stack()
    gen = torch.Generator(device=device).manual_seed(hash(case.name) & 0xFFFF)
    heads = case.num_kv_heads * Q_HEADS_PER_KV
    shape = (case.topk, case.total_q, heads)

    partial_real = torch.randn(*shape, HEAD_DIM, generator=gen, device=device).to(case.partial_dtype)
    # Partial LSEs spread over a few units with a row offset, so the row max sits in a random split.
    lse_partial = torch.randn(*shape, generator=gen, device=device) * 2.0
    lse_partial += torch.randn(case.total_q, heads, generator=gen, device=device) * 4.0
    lse_temperature_partial = (
        lse_partial * 0.7 + torch.randn(*shape, generator=gen, device=device) if case.temperature else None
    )
    if case.split_counts:
        split_counts = torch.randint(1, case.topk + 1, (case.total_q, case.num_kv_heads), generator=gen,
                                     device=device, dtype=torch.int32)
        split_counts[0::7] = case.topk
        split_counts[3::11] = 1
        counts_q = split_counts.repeat_interleave(Q_HEADS_PER_KV, dim=1)
    else:
        split_counts = None
        counts_q = torch.full((case.total_q, heads), case.topk, device=device, dtype=torch.int32)
    valid = torch.arange(case.topk, device=device)[:, None, None] < counts_q[None]
    output_scale = torch.tensor([0.37], dtype=torch.float32, device=device) if case.output_scale else None

    ref_out, ref_lse = _reference(partial_real, lse_partial, valid, output_scale.item() if case.output_scale else 1.0)
    ref_lse_temperature = _reference(partial_real, lse_temperature_partial, valid, 1.0)[1] if case.temperature else None

    # The kernel's inputs: fake column order, and NaN wherever a split is past the row's count.
    o_partial = torch.empty_like(partial_real)
    o_partial[..., _real_to_fake_col(device)] = partial_real
    _nan_fill(o_partial, ~valid)
    lse_in = lse_partial.clone()
    _nan_fill(lse_in, ~valid)
    lse_temperature_in = None
    if case.temperature:
        lse_temperature_in = lse_temperature_partial.clone()
        _nan_fill(lse_temperature_in, ~valid)
    out = torch.empty(case.total_q, heads, HEAD_DIM, dtype=case.out_dtype, device=device)
    lse_out = torch.empty(case.total_q, heads, dtype=torch.float32, device=device) if case.lse_out else None
    lse_temperature_out = torch.empty_like(lse_out) if case.temperature else None

    def run():
        combine(o_partial, lse_in, out, lse_out, lse_temperature_partial=lse_temperature_in,
                lse_temperature_out=lse_temperature_out, split_counts=split_counts,
                output_scale=output_scale, use_pdl=True)
        return out.clone(), None if lse_out is None else lse_out.clone(), (
            None if lse_temperature_out is None else lse_temperature_out.clone())

    first = run_timed(f"combine {case.name}", run)
    second = run()
    torch.cuda.synchronize()

    assert torch.isfinite(first[0]).all(), "a stale partial reached the output"
    # The output is the FP32 merge rounded once to the output dtype.
    torch.testing.assert_close(first[0].float(), ref_out, rtol=1e-2, atol=2e-2)
    if case.lse_out:
        assert torch.isfinite(first[1]).all()
        torch.testing.assert_close(first[1], ref_lse, rtol=1e-5, atol=1e-4)
    if case.temperature:
        torch.testing.assert_close(first[2], ref_lse_temperature, rtol=1e-5, atol=1e-4)
    # Same inputs, same split order: the merge is deterministic.
    for a, b in zip(first, second):
        assert a is None or torch.equal(a, b), f"{case.name}: the combine is not deterministic"
