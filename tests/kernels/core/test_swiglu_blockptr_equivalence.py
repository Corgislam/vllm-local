# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bitwise equivalence tests between the raw-pointer and block-pointer
variants of the SwiGLU-Step Triton kernel.

The two kernels perform the same sequence of FP ops in the same order on
the same dtypes; only the address-generation path differs. Bitwise
equality via `torch.equal` is therefore the primary acceptance check.
A diagnostic max-diff is reported when equality fails.
"""

import pytest
import torch

from vllm.model_executor.layers.activation import (
    SwigluStepAndMul,
    swiglustep_and_mul_triton,
    swiglustep_and_mul_triton_blockptr,
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
LIMITS = [7.0, 1.0, float("inf")]
SEED = 0


def _run_both(B: int, d: int, dtype: torch.dtype, limit: float):
    torch.manual_seed(SEED)
    x = torch.randn(B, 2 * d, dtype=dtype, device="cuda")
    out_raw = torch.empty(B, d, dtype=dtype, device="cuda")
    out_blk = torch.empty(B, d, dtype=dtype, device="cuda")
    swiglustep_and_mul_triton(out_raw, x, limit=limit)
    swiglustep_and_mul_triton_blockptr(out_blk, x, limit=limit)
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
@pytest.mark.parametrize("limit", LIMITS)
def test_swiglu_blockptr_bitwise_equivalence(
    dtype: torch.dtype, shape: tuple[int, int], limit: float
):
    B, d = shape
    out_raw, out_blk = _run_both(B, d, dtype, limit)
    assert torch.equal(out_raw, out_blk), _diag(
        f"B={B} d={d} dtype={dtype} limit={limit}", out_raw, out_blk
    )


def test_swiglu_blockptr_large_shape():
    """One large shape (4096, 14336) in bf16 at limit=7.0.

    Memory: x ~448 MB, two outputs ~224 MB. Well within a 32 GB card.
    """
    B, d = 4096, 14336
    out_raw, out_blk = _run_both(B, d, torch.bfloat16, 7.0)
    assert torch.equal(out_raw, out_blk), _diag(
        f"large B={B} d={d}", out_raw, out_blk
    )


def test_swiglu_blockptr_module_integration():
    """Module-level integration: SwigluStepAndMul.forward_native vs. a
    forward path that dispatches to the block-pointer wrapper.

    This exercises the Python -> kernel call chain used by real models.
    """
    from vllm.config import VllmConfig
    from vllm.config.vllm import set_current_vllm_config

    torch.manual_seed(SEED)
    B, d = 128, 4096
    dtype = torch.bfloat16
    limit = 7.0

    with set_current_vllm_config(VllmConfig()):
        module = SwigluStepAndMul(limit=limit)
    x = torch.randn(B, 2 * d, dtype=dtype, device="cuda")
    out_native = module.forward_native(x)

    out_blk = torch.empty(B, d, dtype=dtype, device="cuda")
    swiglustep_and_mul_triton_blockptr(out_blk, x, limit=limit)

    torch.testing.assert_close(out_blk, out_native, atol=2e-2, rtol=2e-2)
