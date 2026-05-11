# vllm-local — Triton kernel raw-pointer → block-pointer conversion

Local working fork tracking [vllm-project/vllm#40458][rfc] — an RFC to
convert vLLM's core dense-inference Triton kernels from raw pointer
arithmetic (`tl.load(ptr + offset, mask=...)`) to structured block
pointers (`tl.make_block_ptr` / `tl.advance`).

[rfc]: https://github.com/vllm-project/vllm/issues/40458

Original upstream README preserved at [`README.vllm.md`](./README.vllm.md).

## Why

Block pointers:

1. **Enable hardware portability** — tiled-memory accelerators (Intel
   XPU, IBM Spyre/AIU, etc.) cannot lower raw pointer arithmetic.
2. **Leverage Hopper TMA** — block pointers can be lowered to Tensor
   Memory Accelerator instructions on H100/H200.
3. **Align with Triton's direction** — structured memory access is the
   project's primary recommended API.
4. **Improve readability** — memory layout (shape, strides, offsets) is
   separated from computation (load, compute, store).

## Scope of this repo

Local bench and validation work for the five kernels listed in the RFC:

| # | Kernel | File | Status |
|---|--------|------|--------|
| 1 | SwiGLU-Step (`_swiglustep_and_mul_kernel`) | `vllm/model_executor/layers/activation.py` | done on `feat/swiglu-blockptr` |
| 2 | Ranks (`_ranks_kernel`) | `vllm/v1/worker/gpu/sample/logprob.py` | pending |
| 3 | RMSNorm (`_rms_norm_kernel`) | `vllm/model_executor/layers/batch_invariant.py` | pending |
| 4 | Log-softmax (`_topk_log_softmax_kernel`) | `vllm/v1/worker/gpu/sample/logprob.py` | pending |
| 5 | MRoPE (`_triton_mrope_forward`) | `vllm/model_executor/layers/rotary_embedding/mrope.py` | pending |

Each conversion lives on its own branch (`feat/<kernel>-blockptr`) with
a standalone plan, equivalence test, and benchmark script.

Nothing is merged back to `main` inside this repo; it stays as a
collection of parallel feature branches, each a candidate for eventual
upstream submission.

## Branches

### `main`

Pristine snapshot of `vllm-main` plus this README. Baseline for every
feature branch.

### `feat/swiglu-blockptr`

First kernel. Adds `_swiglustep_and_mul_kernel_blockptr` /
`swiglustep_and_mul_triton_blockptr` alongside the original raw-pointer
kernel, with full equivalence tests and a head-to-head benchmark.

See `PLAN-swiglu-blockptr.md` on that branch for the full spec.

Local results (RTX 5090, Triton 3.6.0, CUDA 12.8):

- **Equivalence:** 47/47 `torch.equal` bitwise, across
  fp16 / bf16 / fp32 × five shapes × three clamp limits
- **Performance (bf16, limit=7.0, 7 alternating rounds):**

  | (B, d) | raw median | blk median | speedup |
  | --- | --- | --- | --- |
  | (128, 4096) | 5.63 µs | 5.63 µs | 1.000× |
  | (512, 4096) | 10.27 µs | 10.30 µs | 0.997× |
  | (4096, 4096) | 75.82 µs | 75.30 µs | 1.007× |
  | (4096, 14336) | 241.12 µs | 241.09 µs | 1.000× |

  Speedup range [0.997×, 1.007×] — within measurement noise, no
  regression.

## Reproducing locally

Requires a CUDA GPU, a vLLM source checkout, and the `vllm` conda env
with `torch`, `triton`, `pytest`. Instructions below assume the source
tree has compiled artifacts available somewhere (either built in place
or reused from an installed wheel, see "Environment notes" below).

### Equivalence tests

```bash
conda activate vllm
cd <worktree>
python -m pytest \
    tests/kernels/core/test_swiglu_blockptr_equivalence.py \
    -v --noconftest
```

Expected: `47 passed`.

### Benchmark

```bash
conda activate vllm
cd <worktree>
PYTHONPATH=$(pwd) python benchmarks/kernels/benchmark_swiglu_blockptr.py
```

The script prints GPU / driver / Triton provenance, then one row per
benchmarked shape with raw vs. block-pointer median latency and a
speedup column.

## Environment notes

The `feat/swiglu-blockptr` branch includes a local-dev-only workaround
in `vllm/platforms/cuda.py` that wraps `import vllm._C_stable_libtorch`
in `try/except ModuleNotFoundError`. This is committed as a separate
commit (`chore(local-dev): tolerate missing vllm._C_stable_libtorch`)
and must be reverted before any upstream submission.

If you are reusing compiled artifacts from an installed wheel older
than the current source tree, you may also need to symlink the shared
objects into the source directory:

```bash
SITE=$(python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' \
       2>/dev/null)
for f in _C.abi3.so _flashmla_C.abi3.so _flashmla_extension_C.abi3.so \
         _moe_C.abi3.so cumem_allocator.abi3.so _version.py; do
    ln -sf "$SITE/$f" "vllm/$f"
done
```

These symlinks are not tracked by git (they match `.gitignore`
patterns).

## Layout on each feature branch

```
PLAN-<kernel>-blockptr.md                         # spec + acceptance
benchmarks/kernels/benchmark_<kernel>_blockptr.py # standalone bench
tests/kernels/core/test_<kernel>_blockptr_equivalence.py
vllm/...                                          # kernel + wrapper
```

`main` carries only this README plus the baseline source snapshot; no
kernel changes live on `main`.

## License

vLLM source code is Apache-2.0 (see `LICENSE`). All files added in this
repo carry Apache-2.0 SPDX headers.
