# Plan: Convert SwiGLU-Step Triton kernel from raw pointer to block pointer

Tracking upstream discussion: vllm-project/vllm#40458 (RFC).

Local-only validation, no upstream PR at this stage.

## Goal

Provide a drop-in `tl.make_block_ptr` variant of `_swiglustep_and_mul_kernel`
(together with a matching Python wrapper) and prove, with reproducible data,
that:

1. it is **numerically equivalent** to the existing raw-pointer kernel on
   fp16 / bf16 / fp32 inputs across multiple shapes and `limit` values;
2. it does **not regress performance** on the local GPU (NVIDIA RTX 5090,
   Blackwell sm_120);
3. it works correctly when invoked through the `SwigluStepAndMul` module
   (module-level integration, not full `LLM.generate()` end-to-end).

The original raw-pointer kernel is **kept in place**. The block-pointer
version is added alongside it so the two can be compared in the same process.

## Scope

### In scope

- `vllm/model_executor/layers/activation.py`
  - Add `_swiglustep_and_mul_kernel_blockptr` (Triton kernel, block-pointer
    load/store).
  - Add `swiglustep_and_mul_triton_blockptr` (Python wrapper, same signature
    as existing `swiglustep_and_mul_triton`).
  - Wrapper asserts last-dim contiguity
    (`input.stride(-1) == 1 and output.stride(-1) == 1`).
  - Original `_swiglustep_and_mul_kernel` and `swiglustep_and_mul_triton`
    remain unchanged. `SwigluStepAndMul.forward_cuda` still calls the
    original by default.

- `tests/kernels/core/test_swiglu_blockptr_equivalence.py` (new)
  - Parameterized bitwise equivalence tests against the raw-pointer kernel.
  - One explicit large-shape test.
  - One module-level integration test comparing `SwigluStepAndMul` forward
    path when backed by the block-pointer wrapper vs. `forward_native`.

- `benchmarks/kernels/benchmark_swiglu_blockptr.py` (new)
  - Standalone script (not collected by pytest).
  - In-process head-to-head timing with alternating order per round.

### Out of scope

- Full `vllm.LLM.generate()` end-to-end test (no `SwigluStepAndMul`-using
  model is small enough to fit local storage / VRAM comfortably; Qwen2.5
  / Qwen3 use `SiluAndMul`, not `SwigluStepAndMul`).
- Hopper TMA variant using `tl.make_tensor_descriptor` (RFC open question;
  no Hopper hardware available locally).
- Converting the other four kernels listed in the RFC (RMSNorm, log-softmax,
  Ranks, MRoPE). Those follow after SwiGLU is validated.
- Any upstream PR or change to `SwigluStepAndMul.forward_cuda` dispatch.

## Approach

### Block-pointer design (fixed)

For each program `(i, j)` = `(row, col_block)`, build three independent
block pointers:

```
gate_bp = tl.make_block_ptr(
    base=x_ptr,
    shape=(B, 2 * d),
    strides=(x_stride, 1),
    offsets=(i, j * BLOCK_SIZE),
    block_shape=(1, BLOCK_SIZE),
    order=(1, 0),
)
up_bp = tl.make_block_ptr(
    base=x_ptr,
    shape=(B, 2 * d),
    strides=(x_stride, 1),
    offsets=(i, d + j * BLOCK_SIZE),   # up half starts at column d
    block_shape=(1, BLOCK_SIZE),
    order=(1, 0),
)
out_bp = tl.make_block_ptr(
    base=o_ptr,
    shape=(B, d),
    strides=(o_stride, 1),
    offsets=(i, j * BLOCK_SIZE),
    block_shape=(1, BLOCK_SIZE),
    order=(1, 0),
)
```

Load / store with `boundary_check=(1,)` on the column axis. The row axis
is always in-bounds by construction (grid is `(B, cdiv(d, BLOCK_SIZE))`).

Computation (sigmoid, clamp, multiply, cast) is byte-for-byte identical
to the raw-pointer kernel. Only address generation changes.

`BLOCK_SIZE = 1024`, matching the original. Grid shape unchanged.

### Wrapper invariants

- Same signature: `(output, input, limit=7.0)`.
- Adds `assert input.ndim == 2`, `assert input.size(1) % 2 == 0`,
  `assert input.stride(-1) == 1`, `assert output.stride(-1) == 1`.
- Does **not** register itself as a `CustomOp`; callers (tests, benchmark)
  import it explicitly.

## Validation

### Equivalence test

Location: `tests/kernels/core/test_swiglu_blockptr_equivalence.py`.

Matrix:

- `dtype ∈ {torch.float16, torch.bfloat16, torch.float32}`
- `(B, d) ∈ {(1, 128), (2, 1024), (7, 1000), (17, 1536), (128, 4096)}`
  (mix of block-aligned and non-aligned column counts)
- `limit ∈ {7.0, 1.0, float('inf')}`
- Full Cartesian = 45 cases.

Plus one large-shape test:

- `(B, d) = (4096, 14336)`, `dtype = bfloat16`, `limit = 7.0`.
- Memory budget on a 32 GB 5090 is ~560 MB for all tensors, well within
  headroom.

Plus one module-level integration test:

- Monkey-patch (or subclass) `SwigluStepAndMul` to route `forward_cuda`
  through `swiglustep_and_mul_triton_blockptr`, then compare against
  `forward_native` on one representative shape. Expected close but not
  bitwise (`forward_native` is a pure-PyTorch fp32-then-cast path, not
  an alternate Triton kernel).

Assertions:

- Primary: `torch.equal(out_raw, out_blockptr)` — bitwise identical.
  Rationale: both kernels perform the same sequence of floating-point
  operations in the same order on the same dtypes. Only address generation
  differs, which should not change FP results.
- Fallback when primary fails (diagnostic only, not an acceptance
  criterion): `torch.allclose(out_raw, out_blockptr, atol=0, rtol=0)` with
  maximum absolute and relative differences logged.
- Module integration test: `torch.allclose(out_native, out_blockptr,
  atol=2e-2, rtol=2e-2)` for fp16/bf16.

Fixed seed (`torch.manual_seed(0)`).

### Benchmark

Location: `benchmarks/kernels/benchmark_swiglu_blockptr.py`.

- Shapes: `(B, d) ∈ {(128, 4096), (512, 4096), (4096, 4096), (4096, 14336)}`.
- dtype: `torch.bfloat16` only (kernel is memory-bound; dtype covers the
  same SIMT path).
- `limit = 7.0`.
- Per shape: `N_ROUNDS = 7`. Each round calls `triton.testing.do_bench`
  once for raw and once for block-pointer, with alternating execution
  order (`round % 2 == 0` → raw first; odd → block-pointer first).
- `do_bench(warmup=25, rep=100, quantiles=[0.5])` per invocation.
- Same pre-allocated `x`, `out_raw`, `out_blockptr` tensors throughout a
  shape’s 7 rounds, to keep allocator state frozen.
- Warm up both kernels (3× each) before the first timed round to absorb
  Triton JIT.
- Report per shape: median of 7 round medians for each kernel, plus min
  and max across rounds. Print GPU name, driver, Triton version for
  provenance.
- Not collected by pytest (plain `__main__` entry point).

### Acceptance

- All 45 parameterized cases pass `torch.equal`.
- Large-shape case passes `torch.equal`.
- Module integration case passes `allclose` within the stated tolerance.
- Benchmark shows block-pointer median within ±5 % of raw-pointer median
  on every shape (i.e. no material regression on 5090). Speedups are
  welcome but not required locally.

If bitwise equivalence fails in any case, the work stops and the failure
mode is investigated before any further changes.

## Risks

- **Triton on Blackwell (sm_120):** block-pointer lowering is relatively
  new; regressions or compile failures are possible on bleeding-edge
  targets. Mitigation: the work is in a worktree; failures are isolated
  to the branch.
- **Bitwise equivalence may not hold:** if Triton’s block-pointer path
  performs extra vectorisation that reorders FP adds, the `torch.equal`
  check will fail. Mitigation: fall back to `allclose(atol=0, rtol=0)`
  for diagnosis, record the observed `|Δ|`, and decide whether that
  still counts as acceptable equivalence for this project’s purposes.
- **`SwigluStepAndMul.forward_cuda` not exercised end-to-end:** no
  available local model triggers this kernel. Mitigation: module-level
  integration test covers the Python→kernel call path; any downstream
  end-to-end validation is deferred to when suitable hardware/weights
  are available.
- **Contiguity assumption:** the wrapper asserts last-dim contiguity.
  All call sites in the current codebase already satisfy this (inputs
  come from `nn.Linear` outputs), but the assertion is new. Mitigation:
  documented in the wrapper; callers that violate it get a clear error
  instead of silent corruption.

## Execution order

1. ~~Initialise repo + baseline commit~~ *(done)*.
2. ~~Create worktree `feat/swiglu-blockptr`~~ *(done)*.
3. Write this plan *(in progress)*.
4. Implement kernel + wrapper in `activation.py`.
5. Write equivalence test file.
6. Run equivalence test under `conda activate vllm`.
7. Write benchmark script.
8. Run benchmark and record output.
9. Sediment findings into `~/.agent-knowledge/` (experience +/or skill as
   appropriate).

Every step is independently verifiable. Failures at any step pause
execution pending a diagnosis rather than a forward-fix.
