# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Sparse forward combine kernel and public launcher.

The sparse forward kernels write one partial output and one partial log-sum-exp per
(query, head, selected page) into ``[num_splits, total_q, heads, dim]`` and
``[num_splits, total_q, heads]`` buffers. This kernel merges the splits of every (query, head)
row into the final output and LSE. Rows are independent, so the concatenated varlen rows form
one flat scheduling domain: a CTA owns ``tile_m`` consecutive flat rows, which is one contiguous
chunk of each split's slab, and streams the chunks of its splits through a ring of ``stages``
shared-memory slots with one bulk-tensor copy per (tile, split).

The partial output columns are in the STG.128 "fake" order the forward epilogue stores (a 4x4
transpose of 32-bit pairs inside each 32-column group); the epilogue here undoes it with two
XOR-shuffle stages before one coalesced 128-bit store per row.
"""

import math
from typing import Optional, Type

import cutlass
from cutlass import Float32, Int32, Int64, const_expr
import cutlass.cute as cute
from cutlass.cute import FastDivmodDivisor
from cutlass.cute.nvgpu import cpasync
import cuda.bindings.driver as cuda
import torch

from src.common import utils
from src.common.aot_cache import save_aot, try_load_aot
from src.common.cute_dsl_utils import (
    assume_tensor_aligned,
    compile_with_timing,
    torch2cute_dtype_map,
)

# One tile is 64 flat (query, head) rows of 256 B, a 16 KB chunk per split; four slots keep
# 64 KB in flight per CTA and leave room for three CTAs per SM.
TILE_M = 64
NUM_THREADS = 256
STAGES = 4
# The partial buffer is addressed as (dim, flat row, split); the TMA descriptor lowering is
# relied on for split strides below this many elements.
MAX_SPLIT_SLAB_ELEMENTS = 2**31


class SparseAttentionForwardCombine:
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        dtype_partial: Type[cutlass.Numeric],
        head_dim: int,
        topk: int,
        tile_m: int = TILE_M,
        num_threads: int = NUM_THREADS,
        stages: int = STAGES,
        use_pdl: bool = False,
        min_blocks_per_mp: int = 0,
    ):
        """Forward combine kernel for split attention computation.

        :param dtype: output data type (BF16 or F16)
        :param dtype_partial: partial output data type (BF16 or F16)
        :param head_dim: head dimension; a tile row is one head's ``head_dim`` elements
        :param topk: exact number of split partials per row
        :param tile_m: flat (query, head) rows per CTA
        :param num_threads: threads per CTA
        :param stages: ring slots of ``tile_m`` partial rows each
        :param use_pdl: wait on the producer grid before reading its outputs
        :param min_blocks_per_mp: CTAs per SM ptxas must leave registers for (0 lets it choose)
        """
        if dtype not in (cutlass.BFloat16, cutlass.Float16):
            raise TypeError("the combine writes BF16 or F16 outputs")
        if dtype_partial not in (cutlass.BFloat16, cutlass.Float16):
            raise TypeError("the combine reads BF16 or F16 partials")
        if not 1 <= topk <= 256:
            raise ValueError(f"topk must be in [1, 256], got {topk}")
        if num_threads % 32 != 0 or stages < 1:
            raise ValueError("num_threads must be a multiple of 32 and stages at least 1")
        copy_bits = 128
        self.elems_per_copy = copy_bits // dtype_partial.width
        if head_dim % self.elems_per_copy != 0:
            raise ValueError("head_dim must be a multiple of one 16-byte copy")
        # Threads that share one tile row, 16 bytes of it each.
        self.threads_per_row = head_dim // self.elems_per_copy
        if num_threads % self.threads_per_row != 0:
            raise ValueError("num_threads must be a multiple of the threads per row")
        self.rows_per_pass = num_threads // self.threads_per_row
        if tile_m % self.rows_per_pass != 0:
            raise ValueError("tile_m must be a multiple of the rows one pass covers")
        # Threads that transpose one row of the LSE tile.
        if num_threads % tile_m != 0:
            raise ValueError("num_threads must be a multiple of tile_m")
        self.lse_threads_per_row = num_threads // tile_m
        if 32 % self.lse_threads_per_row != 0 or topk % self.lse_threads_per_row != 0:
            raise ValueError("the threads per LSE row must divide the warp and the split count")
        self.dtype = dtype
        self.dtype_partial = dtype_partial
        self.head_dim = head_dim
        self.topk = topk
        self.tile_m = tile_m
        self.num_threads = num_threads
        self.num_warps = num_threads // 32
        self.stages = stages
        self.use_pdl = use_pdl
        self.min_blocks_per_mp = min_blocks_per_mp

    def _setup_attributes(self):
        copy_bits = 128
        # One thread map serves the reads of a ring slot and the output stores: BF16 and F16
        # have the same width, so a thread's 16 bytes cover the same columns in both.
        thr_layout = cute.make_ordered_layout(
            (self.rows_per_pass, self.threads_per_row), order=(1, 0)
        )
        val_layout = cute.make_layout((1, self.elems_per_copy))
        self.tiled_copy_partial = cute.make_tiled_copy_tv(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.dtype_partial, num_bits_per_copy=copy_bits
            ),
            thr_layout,
            val_layout,
        )
        self.tiled_copy_out = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=copy_bits),
            thr_layout,
            val_layout,
        )
        # LSE tile (split, row): consecutive lanes transpose the splits of one row.
        s2r_thr_layout = cute.make_ordered_layout(
            (self.lse_threads_per_row, self.tile_m), order=(0, 1)
        )
        self.s2r_tiled_copy_lse = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32),
            s2r_thr_layout,
            cute.make_layout(1),
        )
        # The bulk copy writes the tile as a (row, split) box over the same bytes; the trailing
        # mode is the single stage the TMA partition idiom expects.
        self.smem_layout_lse = cute.make_ordered_layout((self.topk, self.tile_m), order=(1, 0))
        self.smem_layout_lse_tma = cute.make_ordered_layout(
            (self.tile_m, self.topk, 1), order=(0, 1, 2)
        )
        # Ring slot: row-major (row, dim); 16 threads read 16 B each of one 256 B row, which is
        # conflict-free without a swizzle. The TMA view is its (dim, row) transpose.
        self.smem_layout_o = cute.make_ordered_layout(
            (self.tile_m, self.head_dim, self.stages), order=(1, 0, 2)
        )
        self.smem_layout_o_tma = cute.make_ordered_layout(
            (self.head_dim, self.tile_m, self.stages), order=(0, 1, 2)
        )

    @cute.jit
    def __call__(
        self,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor] = None,
        mLSE_temperature_partial: Optional[cute.Tensor] = None,
        mLSE_temperature: Optional[cute.Tensor] = None,
        mSplitCounts: Optional[cute.Tensor] = None,
        mOutputScale: Optional[cute.Tensor] = None,
        qhead_per_kvhead: Int32 = Int32(1),
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        if const_expr(mO_partial.element_type != self.dtype_partial):
            raise TypeError("O partial tensor must match dtype_partial")
        if const_expr(mO.element_type != self.dtype):
            raise TypeError("O tensor must match dtype")
        if const_expr(mLSE_partial.element_type != Float32):
            raise TypeError("LSE partial tensor must be Float32")
        if const_expr(mLSE is not None and mLSE.element_type != Float32):
            raise TypeError("LSE tensor must be Float32")
        if const_expr((mLSE_temperature_partial is None) != (mLSE_temperature is None)):
            raise ValueError(
                "temperature LSE partial and output tensors must either both be provided or both be None"
            )
        if const_expr(
            mLSE_temperature_partial is not None
            and (
                mLSE_temperature_partial.element_type != Float32
                or mLSE_temperature.element_type != Float32
            )
        ):
            raise TypeError("temperature LSE tensors must be Float32")
        if const_expr(len(mO_partial.shape) != 4):
            raise ValueError("O partial tensor must be (num_splits, total_q, nheads, headdim)")
        if const_expr(len(mLSE_partial.shape) != 3):
            raise ValueError("LSE partial tensor must be (num_splits, total_q, nheads)")
        if const_expr(len(mO.shape) != 3):
            raise ValueError("O tensor must be (total_q, nheads, headdim)")
        if const_expr(mLSE is not None and len(mLSE.shape) != 2):
            raise ValueError("LSE tensor must be (total_q, nheads)")
        if const_expr(
            mLSE_temperature_partial is not None
            and (len(mLSE_temperature_partial.shape) != 3 or len(mLSE_temperature.shape) != 2)
        ):
            raise ValueError("temperature LSE tensors must match the LSE partial and output shapes")
        if const_expr(mSplitCounts is not None):
            if const_expr(mSplitCounts.element_type != Int32):
                raise TypeError("split_counts tensor must be Int32")
            if const_expr(len(mSplitCounts.shape) != 2):
                raise ValueError("split_counts tensor must be (total_q, nheads_kv)")
        if const_expr(mOutputScale is not None and mOutputScale.element_type != Float32):
            raise TypeError("output_scale tensor must be Float32")

        mO_partial, mO = [assume_tensor_aligned(t) for t in (mO_partial, mO)]
        num_head = mO_partial.shape[2]
        total_flat = mO_partial.shape[1] * num_head
        self._setup_attributes()

        # Contiguous (num_splits, total_q, nheads, headdim) as (headdim, flat row, split): a
        # (headdim, tile_m) box is one tile of one split.
        mO_partial_tma = cute.make_tensor(
            mO_partial.iterator,
            cute.make_layout(
                (self.head_dim, total_flat, self.topk),
                stride=(1, self.head_dim, self.head_dim * total_flat),
            ),
        )
        tma_atom_O_partial, mO_partial_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mO_partial_tma,
            cute.select(self.smem_layout_o_tma, mode=[0, 1]),
            (self.head_dim, self.tile_m),
        )
        # Contiguous (num_splits, total_q, nheads) as (flat row, split): a (tile_m, num_splits)
        # box is a CTA's whole LSE tile.
        mLSE_partial_tma = cute.make_tensor(
            mLSE_partial.iterator,
            cute.make_layout((total_flat, self.topk), stride=(1, total_flat)),
        )
        tma_atom_LSE, mLSE_partial_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mLSE_partial_tma,
            cute.select(self.smem_layout_lse_tma, mode=[0, 1]),
            (self.tile_m, self.topk),
        )
        tma_atom_LSE_temperature = None
        mLSE_temperature_partial_tma = None
        if const_expr(mLSE_temperature_partial is not None):
            mLSE_temperature_partial_tma = cute.make_tensor(
                mLSE_temperature_partial.iterator,
                cute.make_layout((total_flat, self.topk), stride=(1, total_flat)),
            )
            tma_atom_LSE_temperature, mLSE_temperature_partial_tma = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                mLSE_temperature_partial_tma,
                cute.select(self.smem_layout_lse_tma, mode=[0, 1]),
                (self.tile_m, self.topk),
            )

        lse_tile_elems = cute.cosize(self.smem_layout_lse)

        @cute.struct
        class SharedStorage:
            sLSE: cute.struct.Align[cute.struct.MemRange[Float32, lse_tile_elems], 128]
            sLSETemperature: cute.struct.Align[
                cute.struct.MemRange[
                    Float32, lse_tile_elems if mLSE_temperature_partial is not None else 1
                ],
                128,
            ]
            sBlockMaxValidSplit: cute.struct.Align[
                cute.struct.MemRange[Int32, self.num_warps], 128
            ]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.dtype_partial, cute.cosize(self.smem_layout_o)], 128
            ]
            sRingMbar: cute.struct.Align[cute.struct.MemRange[Int64, self.stages], 8]
            sLseMbar: cute.struct.Align[cute.struct.MemRange[Int64, 1], 8]

        grid_dim = (cute.ceil_div(total_flat, self.tile_m), 1, 1)
        self.kernel(
            mO_partial_tma,
            mLSE_partial_tma,
            mLSE_temperature_partial_tma,
            mO,
            mLSE,
            mLSE_temperature,
            mSplitCounts,
            mOutputScale,
            qhead_per_kvhead,
            tma_atom_O_partial,
            tma_atom_LSE,
            tma_atom_LSE_temperature,
            SharedStorage,
            self.smem_layout_lse,
            self.smem_layout_lse_tma,
            self.smem_layout_o,
            self.smem_layout_o_tma,
            self.tiled_copy_partial,
            self.tiled_copy_out,
            self.s2r_tiled_copy_lse,
            FastDivmodDivisor(num_head),
            self.use_pdl,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=self.min_blocks_per_mp,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mO_partial_tma: cute.Tensor,
        mLSE_partial_tma: cute.Tensor,
        mLSE_temperature_partial_tma: Optional[cute.Tensor],
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mLSE_temperature: Optional[cute.Tensor],
        mSplitCounts: Optional[cute.Tensor],
        mOutputScale: Optional[cute.Tensor],
        qhead_per_kvhead: Int32,
        tma_atom_O_partial: cute.CopyAtom,
        tma_atom_LSE: cute.CopyAtom,
        tma_atom_LSE_temperature: Optional[cute.CopyAtom],
        SharedStorage: cutlass.Constexpr,
        smem_layout_lse: cute.Layout,
        smem_layout_lse_tma: cute.Layout,
        smem_layout_o: cute.Layout,
        smem_layout_o_tma: cute.Layout,
        tiled_copy_partial: cute.TiledCopy,
        tiled_copy_out: cute.TiledCopy,
        s2r_tiled_copy_lse: cute.TiledCopy,
        head_divmod: FastDivmodDivisor,
        use_pdl: cutlass.Constexpr[bool],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        m_block, _, _ = cute.arch.block_idx()
        tile_begin = m_block * self.tile_m
        total_flat = mO.shape[0] * mO.shape[1]
        has_temperature = const_expr(mLSE_temperature_partial_tma is not None)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sLSE = storage.sLSE.get_tensor(smem_layout_lse)
        sLSE_tma = storage.sLSE.get_tensor(smem_layout_lse_tma)
        sLSE_temperature = None
        sLSE_temperature_tma = None
        if const_expr(has_temperature):
            sLSE_temperature = storage.sLSETemperature.get_tensor(smem_layout_lse)
            sLSE_temperature_tma = storage.sLSETemperature.get_tensor(smem_layout_lse_tma)
        sBlockMaxValidSplit = storage.sBlockMaxValidSplit.get_tensor((self.num_warps,))
        sO = storage.sO.get_tensor(smem_layout_o)
        sO_tma = storage.sO.get_tensor(smem_layout_o_tma)
        ring_mbar = storage.sRingMbar.data_ptr()
        lse_mbar = storage.sLseMbar.data_ptr()

        # Barriers are CTA-local state: set them up while the producer grid is still draining.
        if tidx == 0:
            for stage in cutlass.range(self.stages, unroll_full=True):
                cute.arch.mbarrier_init(ring_mbar + stage, 1)
            cute.arch.mbarrier_init(lse_mbar, 1)
            cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        if const_expr(use_pdl):
            cute.arch.griddepcontrol_wait()

        # This CTA's tile of every split, and its LSE tile(s).
        gO_partial = cute.local_tile(
            mO_partial_tma, (self.head_dim, self.tile_m), (0, None, None)
        )
        tOsO_tma, tOgO_tma = cpasync.tma_partition(
            tma_atom_O_partial,
            0,
            cute.make_layout(1),
            cute.group_modes(sO_tma, 0, 2),
            cute.group_modes(gO_partial, 0, 2),
        )
        o_tma_bytes = const_expr(
            cute.size_in_bytes(self.dtype_partial, cute.select(smem_layout_o_tma, mode=[0, 1]))
        )
        gLSE = cute.local_tile(mLSE_partial_tma, (self.tile_m, self.topk), (None, 0))
        tLSEsLSE_tma, tLSEgLSE_tma = cpasync.tma_partition(
            tma_atom_LSE,
            0,
            cute.make_layout(1),
            cute.group_modes(sLSE_tma, 0, 2),
            cute.group_modes(gLSE, 0, 2),
        )
        lse_tma_bytes = const_expr(
            cute.size_in_bytes(Float32, cute.select(smem_layout_lse_tma, mode=[0, 1]))
        )
        if const_expr(has_temperature):
            gLSE_temperature = cute.local_tile(
                mLSE_temperature_partial_tma, (self.tile_m, self.topk), (None, 0)
            )
            tLSEsLSE_temperature_tma, tLSEgLSE_temperature_tma = cpasync.tma_partition(
                tma_atom_LSE_temperature,
                0,
                cute.make_layout(1),
                cute.group_modes(sLSE_temperature_tma, 0, 2),
                cute.group_modes(gLSE_temperature, 0, 2),
            )
        primed_stages = const_expr(min(self.stages, self.topk))
        # One warp puts the LSE tile and the first ring slots in flight; they land while the
        # CTA decodes its rows.
        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    lse_mbar, lse_tma_bytes * (2 if has_temperature else 1)
                )
            cute.copy(
                tma_atom_LSE,
                tLSEgLSE_tma[None, m_block],
                tLSEsLSE_tma[None, 0],
                tma_bar_ptr=lse_mbar,
            )
            if const_expr(has_temperature):
                cute.copy(
                    tma_atom_LSE_temperature,
                    tLSEgLSE_temperature_tma[None, m_block],
                    tLSEsLSE_temperature_tma[None, 0],
                    tma_bar_ptr=lse_mbar,
                )
            for stage in cutlass.range(primed_stages, unroll_full=True):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(ring_mbar + stage, o_tma_bytes)
                cute.copy(
                    tma_atom_O_partial,
                    tOgO_tma[None, m_block, Int32(stage)],
                    tOsO_tma[None, stage],
                    tma_bar_ptr=ring_mbar + stage,
                )

        cO = cute.make_identity_tensor((self.tile_m, self.head_dim))
        thr_copy_partial = tiled_copy_partial.get_slice(tidx)
        tOcO = thr_copy_partial.partition_D(cO)
        tOsO = thr_copy_partial.partition_D(sO)
        num_rows = const_expr(cute.size(tOcO, mode=[1]))

        # The LSE tile holds stale values past a row's valid split count and past the last row:
        # fetch each row's count before the wait and mask after the transpose.
        cLSE = cute.make_identity_tensor((self.topk, self.tile_m))
        s2r_thr_copy_lse = s2r_tiled_copy_lse.get_slice(tidx)
        ts2rsLSE = s2r_thr_copy_lse.partition_S(sLSE)
        ts2rcLSE = s2r_thr_copy_lse.partition_S(cLSE)
        ts2rrLSE = cute.make_rmem_tensor_like(ts2rsLSE)
        lse_rows = const_expr(cute.size(ts2rrLSE, mode=[2]))
        lse_splits = const_expr(cute.size(ts2rrLSE, mode=[1]))
        lse_row_count = cute.make_rmem_tensor(lse_rows, Int32)
        for m in cutlass.range(lse_rows, unroll_full=True):
            row_idx = tile_begin + ts2rcLSE[0, 0, m][1]
            lse_row_count[m] = Int32(0)
            if row_idx < total_flat:
                if const_expr(mSplitCounts is not None):
                    row_q, row_h = divmod(row_idx, head_divmod)
                    lse_row_count[m] = mSplitCounts[row_q, row_h // qhead_per_kvhead]
                else:
                    lse_row_count[m] = Int32(self.topk)
        cute.arch.mbarrier_wait(lse_mbar, 0)
        cute.copy(s2r_tiled_copy_lse, ts2rsLSE, ts2rrLSE)
        if const_expr(has_temperature):
            ts2rsLSE_temperature = s2r_thr_copy_lse.partition_S(sLSE_temperature)
            ts2rrLSE_temperature = cute.make_rmem_tensor_like(ts2rsLSE_temperature)
            cute.copy(s2r_tiled_copy_lse, ts2rsLSE_temperature, ts2rrLSE_temperature)
        for m in cutlass.range(lse_rows, unroll_full=True):
            for sp in cutlass.range(lse_splits, unroll_full=True):
                if ts2rcLSE[0, sp, 0][0] >= lse_row_count[m]:
                    ts2rrLSE[0, sp, m] = -Float32.inf
                    if const_expr(has_temperature):
                        ts2rrLSE_temperature[0, sp, m] = -Float32.inf

        # Per row: the max LSE over the valid splits, the split weights exp(lse - max) normalized
        # by their sum (written back over the LSE tile), the final LSE, and the last valid split.
        final_lse = cute.make_rmem_tensor(lse_rows, Float32)
        max_valid_split = cute.make_rmem_tensor(lse_rows, Int32)
        threads_per_row = const_expr(self.lse_threads_per_row)
        LOG2_E = math.log2(math.e)
        for m in cutlass.range(lse_rows, unroll_full=True):
            lse_max = cute.arch.warp_reduction_max(
                ts2rrLSE[None, None, m]
                .load()
                .reduce(cute.ReductionOp.MAX, init_val=-Float32.inf, reduction_profile=0),
                threads_in_group=threads_per_row,
            )
            max_valid_idx = -1
            for sp in cutlass.range(lse_splits, unroll_full=True):
                if ts2rrLSE[0, sp, m] != -Float32.inf:
                    max_valid_idx = ts2rcLSE[0, sp, 0][0]
            max_valid_split[m] = cute.arch.warp_reduction_max(
                max_valid_idx, threads_in_group=threads_per_row
            )
            # A row without valid splits has max -inf; its weights are exp2(-inf) = 0.
            lse_max_cur = 0.0 if lse_max == -Float32.inf else lse_max
            lse_sum_cur = 0.0
            for sp in cutlass.range(lse_splits, unroll_full=True):
                scale = cute.math.exp2(
                    ts2rrLSE[0, sp, m] * LOG2_E - (lse_max_cur * LOG2_E), fastmath=True
                )
                lse_sum_cur += scale
                ts2rrLSE[0, sp, m] = scale
            lse_sum_cur = cute.arch.warp_reduction_sum(
                lse_sum_cur, threads_in_group=threads_per_row
            )
            inv_sum = 0.0
            if max_valid_split[m] < 0 or lse_sum_cur == 0.0 or lse_sum_cur != lse_sum_cur:
                final_lse[m] = -Float32.inf
            else:
                final_lse[m] = cute.math.log(lse_sum_cur, fastmath=True) + lse_max
                inv_sum = 1.0 / lse_sum_cur
            ts2rrLSE[None, None, m].store(ts2rrLSE[None, None, m].load() * inv_sum)
        cute.copy(s2r_tiled_copy_lse, ts2rrLSE, ts2rsLSE)

        if const_expr(has_temperature):
            final_lse_temperature = cute.make_rmem_tensor(lse_rows, Float32)
            for m in cutlass.range(lse_rows, unroll_full=True):
                lse_temperature_max = cute.arch.warp_reduction_max(
                    ts2rrLSE_temperature[None, None, m]
                    .load()
                    .reduce(cute.ReductionOp.MAX, init_val=-Float32.inf, reduction_profile=0),
                    threads_in_group=threads_per_row,
                )
                lse_temperature_max_cur = (
                    0.0 if lse_temperature_max == -Float32.inf else lse_temperature_max
                )
                lse_temperature_sum_cur = 0.0
                for sp in cutlass.range(lse_splits, unroll_full=True):
                    lse_temperature_sum_cur += cute.math.exp2(
                        ts2rrLSE_temperature[0, sp, m] * LOG2_E
                        - (lse_temperature_max_cur * LOG2_E),
                        fastmath=True,
                    )
                lse_temperature_sum_cur = cute.arch.warp_reduction_sum(
                    lse_temperature_sum_cur, threads_in_group=threads_per_row
                )
                if (
                    max_valid_split[m] < 0
                    or lse_temperature_sum_cur == 0.0
                    or lse_temperature_sum_cur != lse_temperature_sum_cur
                ):
                    final_lse_temperature[m] = -Float32.inf
                else:
                    final_lse_temperature[m] = (
                        cute.math.log(lse_temperature_sum_cur, fastmath=True)
                        + lse_temperature_max
                    )

        # The ring is shared by the CTA, so the split loop runs to the tile's deepest valid
        # split: one slot per warp, reduced after the barrier below.
        thread_max_valid_split = max_valid_split[0]
        for m in cutlass.range(1, lse_rows, unroll_full=True):
            thread_max_valid_split = max(thread_max_valid_split, max_valid_split[m])
        warp_max_valid_split = cute.arch.warp_reduction_max(
            thread_max_valid_split, threads_in_group=32
        )
        if tidx % 32 == 0:
            sBlockMaxValidSplit[tidx // 32] = warp_max_valid_split

        if const_expr(mLSE is not None):
            for m in cutlass.range(lse_rows, unroll_full=True):
                if ts2rcLSE[0, 0, m][0] == 0:  # the lane holding split 0 writes the row
                    row_idx = tile_begin + ts2rcLSE[0, 0, m][1]
                    if row_idx < total_flat:
                        row_q, row_h = divmod(row_idx, head_divmod)
                        mLSE[row_q, row_h] = final_lse[m]
                        if const_expr(mLSE_temperature is not None):
                            mLSE_temperature[row_q, row_h] = final_lse_temperature[m]

        cute.arch.sync_threads()
        block_max_valid_split = sBlockMaxValidSplit[0]
        for wi in cutlass.range(1, self.num_warps, unroll_full=True):
            block_max_valid_split = max(block_max_valid_split, sBlockMaxValidSplit[wi])

        # Accumulate the weighted partials in FP32. A row's weight is zero for its invalid splits
        # and for every split of a row past the last one, so stale partial rows never reach the
        # sum.
        tOrO_partial = cute.make_rmem_tensor_like(tOsO[None, None, None, 0])
        tOrO = cute.make_rmem_tensor_like(tOrO_partial, Float32)
        tOrO.fill(0.0)
        for s in cutlass.range(block_max_valid_split + 1, unroll=2):
            stage = s % Int32(self.stages)
            phase = (s // Int32(self.stages)) & Int32(1)
            scale = cute.make_rmem_tensor(num_rows, Float32)
            for m in cutlass.range(num_rows, unroll_full=True):
                scale[m] = sLSE[s, tOcO[0, m, 0][0]]
            cute.arch.mbarrier_wait(ring_mbar + stage, phase)
            cute.autovec_copy(tOsO[None, None, None, stage], tOrO_partial)
            for m in cutlass.range(num_rows, unroll_full=True):
                if scale[m] > 0.0:
                    tOrO[None, m, None].store(
                        tOrO[None, m, None].load()
                        + scale[m] * tOrO_partial[None, m, None].load().to(Float32)
                    )
            # Every thread has copied this slot to registers before it is refilled.
            cute.arch.sync_threads()
            split_to_load = s + Int32(self.stages)
            if warp_idx == 0 and split_to_load <= block_max_valid_split:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(ring_mbar + stage, o_tma_bytes)
                cute.copy(
                    tma_atom_O_partial,
                    tOgO_tma[None, m_block, split_to_load],
                    tOsO_tma[None, stage],
                    tma_bar_ptr=ring_mbar + stage,
                )

        # Primed slots the split loop did not consume still have a bulk copy in flight; wait for
        # them so none lands after the CTA has exited.
        for stage in cutlass.range(primed_stages, unroll_full=True):
            if Int32(stage) > block_max_valid_split:
                cute.arch.mbarrier_wait(ring_mbar + stage, 0)

        # The partial columns are a 4x4 transpose of 32-bit pairs inside each 32-column group:
        # undo it with two XOR-shuffle stages across the four lanes that share the group, then
        # keep the coalesced 128-bit store. No shared memory, no barrier.
        output_scale = Float32(1.0)
        if const_expr(mOutputScale is not None):
            output_scale = mOutputScale[0]
        thr_copy_out = tiled_copy_out.get_slice(tidx)
        tOcO_store = thr_copy_out.partition_D(cO)
        rO = cute.make_rmem_tensor(tOcO_store.shape, self.dtype)
        elems_per_store = const_expr(cute.size(tiled_copy_out.layout_tv_tiled[1]))
        num_vals = const_expr(cute.size(tOcO, mode=[0]))
        assert num_vals == 8 and cute.size(tOcO_store, mode=[1]) == num_rows
        lane_in_pair_group = tidx % Int32(4)
        for m in cutlass.range(num_rows, unroll_full=True):
            for k in cutlass.range(cute.size(tOcO_store, mode=[2]), unroll_full=True):
                o0, o1 = tOrO[0, m, k], tOrO[1, m, k]
                o2, o3 = tOrO[2, m, k], tOrO[3, m, k]
                o4, o5 = tOrO[4, m, k], tOrO[5, m, k]
                o6, o7 = tOrO[6, m, k], tOrO[7, m, k]
                if const_expr(mOutputScale is not None):
                    o0, o1 = cute.arch.mul_packed_f32x2((o0, o1), (output_scale, output_scale))
                    o2, o3 = cute.arch.mul_packed_f32x2((o2, o3), (output_scale, output_scale))
                    o4, o5 = cute.arch.mul_packed_f32x2((o4, o5), (output_scale, output_scale))
                    o6, o7 = cute.arch.mul_packed_f32x2((o6, o7), (output_scale, output_scale))
                word0 = utils.cvt_f16x2_f32(o0, o1, self.dtype)
                word1 = utils.cvt_f16x2_f32(o2, o3, self.dtype)
                word2 = utils.cvt_f16x2_f32(o4, o5, self.dtype)
                word3 = utils.cvt_f16x2_f32(o6, o7, self.dtype)

                partner0 = cute.arch.shuffle_sync_bfly(word1, offset=1)
                partner1 = cute.arch.shuffle_sync_bfly(word0, offset=1)
                partner2 = cute.arch.shuffle_sync_bfly(word3, offset=1)
                partner3 = cute.arch.shuffle_sync_bfly(word2, offset=1)
                stage0, stage1 = word0, word1
                stage2, stage3 = word2, word3
                if lane_in_pair_group % Int32(2) != Int32(0):
                    stage0, stage2 = partner0, partner2
                if lane_in_pair_group % Int32(2) != Int32(1):
                    stage1, stage3 = partner1, partner3

                partner0 = cute.arch.shuffle_sync_bfly(stage2, offset=2)
                partner1 = cute.arch.shuffle_sync_bfly(stage3, offset=2)
                partner2 = cute.arch.shuffle_sync_bfly(stage0, offset=2)
                partner3 = cute.arch.shuffle_sync_bfly(stage1, offset=2)
                out0, out1 = stage0, stage1
                out2, out3 = stage2, stage3
                if lane_in_pair_group // Int32(2) != Int32(0):
                    out0, out1 = partner0, partner1
                if lane_in_pair_group // Int32(2) != Int32(1):
                    out2, out3 = partner2, partner3

                rO_pairs = cute.recast_tensor(rO[None, m, k], Int32)
                rO_pairs[0] = out0
                rO_pairs[1] = out1
                rO_pairs[2] = out2
                rO_pairs[3] = out3

        for m in cutlass.range(num_rows, unroll_full=True):
            idx = tile_begin + tOcO_store[0, m, 0][0]
            if idx < total_flat:
                row_q, row_h = divmod(idx, head_divmod)
                mO_row = cute.tiled_divide(mO[row_q, row_h, None], (elems_per_store,))
                for k in cutlass.range(cute.size(tOcO_store, mode=[2]), unroll_full=True):
                    k_idx = tOcO_store[0, 0, k][1] // elems_per_store
                    cute.copy(thr_copy_out, rO[None, m, k], mO_row[None, k_idx])



_combine_compile_cache = {}


def _cutlass_dtype(torch_dtype: torch.dtype):
    if torch_dtype not in torch2cute_dtype_map:
        raise TypeError(f"Unsupported dtype: {torch_dtype}")
    return torch2cute_dtype_map[torch_dtype]


def _check_cuda_contiguous(tensor: torch.Tensor, *, name: str, device: torch.device) -> None:
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.data_ptr() % 16:
        raise ValueError(f"{name} must be 16-byte aligned")


def combine(
    o_partial_fake,
    lse_partial,
    o_out,
    lse_out,
    *,
    lse_temperature_partial=None,
    lse_temperature_out=None,
    split_counts=None,
    output_scale=None,
    use_pdl=False,
):
    """K2: merge the sparse forward's split partials into the final output.

    Every (query, head) row is combined on its own, so the tensors use the varlen (``total_q``)
    layout whatever the batch structure. When ``lse_out`` is given, the kernel writes the final
    log-sum-exp of each row into it.

    Args:
        o_partial_fake: [num_splits, total_q, head_q, 128] BF16 or F16, contiguous, columns in
            the forward epilogue's STG.128 order.
        lse_partial: [num_splits, total_q, head_q] FP32, contiguous.
        o_out: [total_q, head_q, 128] BF16 or F16, contiguous.
        lse_out: [total_q, head_q] FP32, or None.
        lse_temperature_partial, lse_temperature_out: optional temperature-scaled LSE partials
            and output shaped like ``lse_partial`` and ``lse_out``; both or neither.
        split_counts: optional [total_q, head_kv] int32 valid split count per row; splits at or
            past a row's count are ignored. ``head_q`` must be a multiple of ``head_kv``.
        output_scale: optional FP32 tensor with at least one element; its first element scales
            the output once before the store.
        use_pdl: wait on the producer grid (programmatic dependent launch) before reading.
    """
    if o_partial_fake.ndim != 4:
        raise ValueError(
            f"o_partial_fake must be [num_splits, total_q, head_q, dim], got {tuple(o_partial_fake.shape)}"
        )
    num_splits, total_q, head_q, dim = o_partial_fake.shape
    if dim != 128:
        raise ValueError(f"the combine supports head dim 128, got {dim}")
    if not 1 <= num_splits <= 256:
        raise ValueError(f"num_splits must be in [1, 256], got {num_splits}")
    if head_q % 4:
        raise ValueError(f"head_q must be a multiple of 4, got {head_q}")
    if total_q * head_q * dim >= MAX_SPLIT_SLAB_ELEMENTS:
        raise ValueError(
            f"total_q * head_q * dim must be below {MAX_SPLIT_SLAB_ELEMENTS} elements per split"
        )
    if o_partial_fake.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(
            f"o_partial_fake dtype must be torch.bfloat16 or torch.float16, got {o_partial_fake.dtype}"
        )
    if o_out.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"o_out dtype must be torch.bfloat16 or torch.float16, got {o_out.dtype}")
    if tuple(o_out.shape) != (total_q, head_q, dim):
        raise ValueError(
            f"o_out must have shape {(total_q, head_q, dim)}, got {tuple(o_out.shape)}"
        )
    if lse_partial.dtype != torch.float32:
        raise TypeError(f"lse_partial must be torch.float32, got {lse_partial.dtype}")
    if tuple(lse_partial.shape) != (num_splits, total_q, head_q):
        raise ValueError(
            f"lse_partial must have shape {(num_splits, total_q, head_q)}, got {tuple(lse_partial.shape)}"
        )
    if lse_out is not None:
        if lse_out.dtype != torch.float32:
            raise TypeError(f"lse_out must be torch.float32, got {lse_out.dtype}")
        if tuple(lse_out.shape) != (total_q, head_q):
            raise ValueError(
                f"lse_out must have shape {(total_q, head_q)}, got {tuple(lse_out.shape)}"
            )
    if (lse_temperature_partial is None) != (lse_temperature_out is None):
        raise ValueError(
            "lse_temperature_partial and lse_temperature_out must either both be provided or both be None"
        )
    if lse_temperature_out is not None:
        if lse_out is None:
            raise ValueError("lse_temperature_out requires lse_out")
        if lse_temperature_partial.shape != lse_partial.shape:
            raise ValueError(
                "lse_temperature_partial must have the same shape as lse_partial, "
                f"got {tuple(lse_temperature_partial.shape)} vs {tuple(lse_partial.shape)}"
            )
        if lse_temperature_out.shape != lse_out.shape:
            raise ValueError(
                "lse_temperature_out must have the same shape as lse_out, "
                f"got {tuple(lse_temperature_out.shape)} vs {tuple(lse_out.shape)}"
            )
        if lse_temperature_out.dtype != torch.float32 or lse_temperature_partial.dtype != torch.float32:
            raise TypeError("temperature LSE tensors must be torch.float32")
    if split_counts is not None:
        if split_counts.dtype != torch.int32:
            raise TypeError(f"split_counts must be torch.int32, got {split_counts.dtype}")
        if split_counts.ndim != 2 or split_counts.shape[0] != total_q:
            raise ValueError(
                f"split_counts must have shape [total_q, head_kv], got {tuple(split_counts.shape)}"
            )
        if head_q % split_counts.shape[1] != 0:
            raise ValueError(
                f"head_q ({head_q}) must be divisible by split_counts heads ({split_counts.shape[1]})"
            )
    if output_scale is not None:
        if output_scale.dtype != torch.float32:
            raise TypeError(f"output_scale must be torch.float32, got {output_scale.dtype}")
        if output_scale.numel() < 1:
            raise ValueError("output_scale must contain at least one element")
    device = o_out.device
    if not o_out.is_cuda:
        raise ValueError("o_out must be a CUDA tensor")
    for name, tensor in (
        ("o_partial_fake", o_partial_fake),
        ("lse_partial", lse_partial),
        ("o_out", o_out),
        ("lse_out", lse_out),
        ("lse_temperature_partial", lse_temperature_partial),
        ("lse_temperature_out", lse_temperature_out),
        ("split_counts", split_counts),
        ("output_scale", output_scale),
    ):
        if tensor is not None:
            _check_cuda_contiguous(tensor, name=name, device=device)

    partial_dtype = _cutlass_dtype(o_partial_fake.dtype)
    out_dtype = _cutlass_dtype(o_out.dtype)
    has_lse = lse_out is not None
    has_temperature = lse_temperature_out is not None
    has_split_counts = split_counts is not None
    has_output_scale = output_scale is not None
    qhead_per_kvhead = head_q // split_counts.shape[1] if has_split_counts else 1
    capability = torch.cuda.get_device_capability(device)

    key = (
        "combine",
        capability,
        dim,
        TILE_M,
        NUM_THREADS,
        STAGES,
        num_splits,
        partial_dtype,
        out_dtype,
        has_lse,
        has_temperature,
        has_split_counts,
        has_output_scale,
        use_pdl,
    )
    if key not in _combine_compile_cache:
        loaded = try_load_aot(key)
        if loaded is not None:
            _combine_compile_cache[key] = loaded
        else:
            from quack.compile_utils import make_fake_tensor

            kernel = SparseAttentionForwardCombine(
                dtype=out_dtype,
                dtype_partial=partial_dtype,
                head_dim=dim,
                topk=num_splits,
                tile_m=TILE_M,
                num_threads=NUM_THREADS,
                stages=STAGES,
                use_pdl=use_pdl,
            )
            total_q_sym, nheads_sym = (cute.sym_int64() for _ in range(2))
            mO_partial = make_fake_tensor(
                partial_dtype,
                (num_splits, total_q_sym, nheads_sym, dim),
                divisibility=128 // partial_dtype.width,
            )
            mLSE_partial = make_fake_tensor(
                Float32, (num_splits, total_q_sym, nheads_sym), divisibility=1, leading_dim=2
            )
            mO = make_fake_tensor(
                out_dtype, (total_q_sym, nheads_sym, dim), divisibility=128 // out_dtype.width
            )
            mLSE = (
                make_fake_tensor(Float32, (total_q_sym, nheads_sym), divisibility=1, leading_dim=1)
                if has_lse
                else None
            )
            mLSE_temperature_partial = (
                make_fake_tensor(
                    Float32, (num_splits, total_q_sym, nheads_sym), divisibility=1, leading_dim=2
                )
                if has_temperature
                else None
            )
            mLSE_temperature = (
                make_fake_tensor(Float32, (total_q_sym, nheads_sym), divisibility=1, leading_dim=1)
                if has_temperature
                else None
            )
            mSplitCounts = (
                make_fake_tensor(
                    Int32, (cute.sym_int64(), cute.sym_int64()), divisibility=1, leading_dim=1
                )
                if has_split_counts
                else None
            )
            mOutputScale = (
                make_fake_tensor(Float32, (cute.sym_int64(),), divisibility=1, leading_dim=0)
                if has_output_scale
                else None
            )
            stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
            _combine_compile_cache[key] = compile_with_timing(
                kernel,
                mO_partial,
                mLSE_partial,
                mO,
                mLSE,
                mLSE_temperature_partial,
                mLSE_temperature,
                mSplitCounts,
                mOutputScale,
                Int32(qhead_per_kvhead),
                stream,
                options="--enable-tvm-ffi",
            )
            save_aot(key, _combine_compile_cache[key], sources=[__name__])

    with torch.cuda.nvtx.range("K2_Combine"):
        _combine_compile_cache[key](
            o_partial_fake,
            lse_partial,
            o_out,
            lse_out,
            lse_temperature_partial,
            lse_temperature_out,
            split_counts,
            output_scale,
            qhead_per_kvhead,
        )
