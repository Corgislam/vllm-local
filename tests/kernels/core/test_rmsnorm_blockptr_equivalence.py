# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Equivalence tests between the raw-pointer and block-pointer variants
of the RMSNorm Triton kernel.

fp16 and bfloat16 are asserted bitwise: the fp32 accumulator's bottom
~13 bits are zero when stored at half precision, so the output is
insensitive to the reduction order that Triton's block-pointer
lowering uses.

fp32 is asserted with a tight absolute tolerance (`atol=2e-6`).
Triton's `tl.load(block_ptr)` and `tl.load(ptr + arange, mask)` lower
to different vectorisation patterns, which yields different reduction
trees and ~1-2 ulp drift on fp32 accumulators. This is a property of
the lowering, not the kernel logic, and is the accepted outcome for
reduction kernels (flash-attention and similar kernels apply the same
tolerance).
"""

import pytest
import torch

from vllm.model_executor.layers.batch_invariant import (
    rms_norm,
    rms_norm_blockptr,
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
EPS_VALUES = [1e-6, 1e-5, 1e-4]
SEED = 0


def _run_both(n_rows: int, n_cols: int, dtype: torch.dtype, eps: float):
    torch.manual_seed(SEED)
    x = torch.randn(n_rows, n_cols, dtype=dtype, device="cuda")
    w = torch.randn(n_cols, dtype=dtype, device="cuda")
    out_raw = rms_norm(x, w, eps=eps)
    out_blk = rms_norm_blockptr(x, w, eps=eps)
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
@pytest.mark.parametrize("eps", EPS_VALUES)
def test_rmsnorm_blockptr_bitwise_equivalence(
    dtype: torch.dtype, shape: tuple[int, int], eps: float
):
    n_rows, n_cols = shape
    out_raw, out_blk = _run_both(n_rows, n_cols, dtype, eps)
    if dtype == torch.float32:
        # See module docstring: fp32 drifts ~1-2 ulp under the block-pointer
        # lowering because the reduction tree differs from raw-pointer load.
        torch.testing.assert_close(out_blk, out_raw, atol=2e-6, rtol=0)
    else:
        assert torch.equal(out_raw, out_blk), _diag(
            f"rows={n_rows} cols={n_cols} dtype={dtype} eps={eps}",
            out_raw,
            out_blk,
        )


def test_rmsnorm_blockptr_large_shape():
    """One large shape (4096, 14336) in bf16 at eps=1e-6.

    Memory: x ~112 MB, weight ~28 KB, two outputs ~224 MB. Well within
    a 32 GB card.
    """
    n_rows, n_cols = 4096, 14336
    out_raw, out_blk = _run_both(n_rows, n_cols, torch.bfloat16, 1e-6)
    assert torch.equal(out_raw, out_blk), _diag(
        f"large rows={n_rows} cols={n_cols}", out_raw, out_blk
    )


def _rmsnorm_reference(
    x: torch.Tensor, w: torch.Tensor, eps: float
) -> torch.Tensor:
    """Pure-PyTorch fp32-accum RMSNorm reference."""
    dtype = x.dtype
    x_f32 = x.float()
    mean_sq = x_f32.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(mean_sq + eps)
    return (x_f32 * inv_rms * w.float()).to(dtype)


def test_rmsnorm_blockptr_reference_integration():
    """Compare block-pointer wrapper output against a pure-PyTorch
    fp32-accum reference. Not bitwise: the reference runs the whole
    expression in fp32 end-to-end, whereas the kernel mixes fp32 accum
    with load-dtype intermediate rounding. allclose with a generous
    tolerance suffices.
    """
    torch.manual_seed(SEED)
    n_rows, n_cols = 128, 4096
    dtype = torch.bfloat16
    eps = 1e-6
    x = torch.randn(n_rows, n_cols, dtype=dtype, device="cuda")
    w = torch.randn(n_cols, dtype=dtype, device="cuda")

    out_ref = _rmsnorm_reference(x, w, eps)
    out_blk = rms_norm_blockptr(x, w, eps=eps)

    torch.testing.assert_close(out_blk, out_ref, atol=2e-2, rtol=2e-2)
