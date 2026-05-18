# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Equivalence tests between the raw-pointer and block-pointer variants
of the batch-invariant log_softmax Triton kernel.

The primary acceptance check is raw-pointer vs. block-pointer equivalence.
The block-pointer kernel preserves the raw kernel's three-pass FP op order
and [BLOCK_SIZE] reduction shape. It keeps a logical mask for the max and
sum-exp passes because block-pointer boundary padding cannot express
other=-inf and padded columns must not contribute to sum_exp.
"""

import pytest
import torch

from vllm.model_executor.layers.batch_invariant import (
    log_softmax,
    log_softmax_blockptr,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires CUDA"
)

DTYPES = [torch.float16, torch.bfloat16, torch.float32]
SHAPES = [
    (1, 128),      # tiny, single-block
    (2, 1024),     # block-aligned
    (7, 1000),     # non-aligned, triggers boundary_check
    (17, 1536),    # non-aligned across two blocks
    (128, 4096),   # medium, multi-block
]
SEED = 0


def _run_both(n_rows: int, n_cols: int, dtype: torch.dtype):
    torch.manual_seed(SEED)
    x = torch.randn(n_rows, n_cols, dtype=dtype, device="cuda")
    out_raw = log_softmax(x, dim=-1)
    out_blk = log_softmax_blockptr(x, dim=-1)
    return out_raw, out_blk


def _diag(name: str, out_raw: torch.Tensor, out_blk: torch.Tensor) -> str:
    diff = (out_raw.float() - out_blk.float()).abs()
    return (
        f"{name}: not bitwise equal. "
        f"max|Δ|={diff.max().item():.3e}, "
        f"mean|Δ|={diff.mean().item():.3e}, "
        f"mismatched={(out_raw != out_blk).sum().item()} / {out_raw.numel()}"
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_logsoftmax_blockptr_equivalence(
    dtype: torch.dtype, shape: tuple[int, int]
):
    n_rows, n_cols = shape
    out_raw, out_blk = _run_both(n_rows, n_cols, dtype)
    if dtype == torch.float32:
        torch.testing.assert_close(out_blk, out_raw, atol=2e-6, rtol=0)
    else:
        assert torch.equal(out_raw, out_blk), _diag(
            f"rows={n_rows} cols={n_cols} dtype={dtype}", out_raw, out_blk
        )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n_cols", [1, 7, 1025, 2049])
def test_logsoftmax_blockptr_all_negative_padding(dtype: torch.dtype, n_cols: int):
    """Regression guard for max-pass padding.

    boundary_check padding only supports zero/nan. The block-pointer kernel
    must remask padded lanes to -inf in the max pass; otherwise all-negative
    rows with a ragged final block would incorrectly use 0 as the row max.
    """
    torch.manual_seed(SEED)
    n_rows = 7
    x = -torch.rand(n_rows, n_cols, dtype=dtype, device="cuda") - 1.0
    out_raw = log_softmax(x, dim=-1)
    out_blk = log_softmax_blockptr(x, dim=-1)
    if dtype == torch.float32:
        torch.testing.assert_close(out_blk, out_raw, atol=2e-6, rtol=0)
    else:
        assert torch.equal(out_raw, out_blk), _diag(
            f"all-negative cols={n_cols} dtype={dtype}", out_raw, out_blk
        )


def test_logsoftmax_blockptr_large_shape():
    """One large shape (4096, 14336) in bf16.

    Memory: x ~112 MB, two outputs ~224 MB. Well within a 32 GB card.
    """
    n_rows, n_cols = 4096, 14336
    out_raw, out_blk = _run_both(n_rows, n_cols, torch.bfloat16)
    assert torch.equal(out_raw, out_blk), _diag(
        f"large rows={n_rows} cols={n_cols}", out_raw, out_blk
    )


def test_logsoftmax_blockptr_reference_integration():
    """Compare block-pointer wrapper output against PyTorch log_softmax.

    This is a sanity integration check, not the primary acceptance criterion.
    The migration target is raw-pointer vs. block-pointer equivalence.
    """
    torch.manual_seed(SEED)
    n_rows, n_cols = 128, 4096
    dtype = torch.bfloat16
    x = torch.randn(n_rows, n_cols, dtype=dtype, device="cuda")

    out_ref = torch.log_softmax(x, dim=-1)
    out_blk = log_softmax_blockptr(x, dim=-1)

    torch.testing.assert_close(out_blk, out_ref, atol=2e-2, rtol=2e-2)
