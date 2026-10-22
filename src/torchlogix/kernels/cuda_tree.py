"""Fused per-tree LUT kernels in CUDA C++ (Stage 3, bake-off candidate "cuda").

Same contract as `triton_tree` (see there): one thread = one (sample, pooled position)
for the block's kernel index k; the 2^D leaves and all tree levels live in registers; only
the root is written; backward recomputes the interior, reduces grad_W with warp shuffles
(every lane of a warp shares k) and one atomicAdd per warp, and adds grad_x atomically.
Built with --fmad=false so the fp32 forward keeps the reference operation order.

JIT-compiled on first use with torch.utils.cpp_extension.load_inline (needs nvcc).
"""

_CPP = r"""
#include <torch/extension.h>
std::vector<torch::Tensor> fwd(torch::Tensor x, torch::Tensor leaf, torch::Tensor pos,
                               torch::Tensor w, int64_t depth, bool out_bf16, bool store_arg);
std::vector<torch::Tensor> bwd(torch::Tensor gy, torch::Tensor x, torch::Tensor leaf,
                               torch::Tensor pos, torch::Tensor arg, torch::Tensor w,
                               int64_t depth, bool need_gx);
"""

_CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <type_traits>
#include <math.h>

__device__ __forceinline__ float ld(const float* p) { return *p; }
__device__ __forceinline__ float ld(const uint8_t* p) { return (float)(*p); }
__device__ __forceinline__ float ld(const __nv_bfloat16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ void st(float* p, float v) { *p = v; }
__device__ __forceinline__ void st(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }

__device__ __forceinline__ float lut(float a, float b, float w0, float w1, float w2, float w3) {
  float t0 = w0 * ((1.0f - a) * (1.0f - b));
  float t1 = w1 * ((1.0f - a) * b);
  float t2 = w2 * (a * (1.0f - b));
  float t3 = w3 * (a * b);
  return ((t0 + t1) + t2) + t3;
}

template <int D> struct Dims {
  static constexpr int S0 = 1 << (D - 1);
};

__device__ __forceinline__ long level_off(int S0, int lvl, int K) {
  return (long)(2 * S0 - 2 * (S0 >> lvl)) * K * 4;
}

// root of the tree at (b, k, p); v[] is scratch of size S0
template <typename TX, int D>
__device__ __forceinline__ float tree_root(const TX* xb, const int* lf, const float* w,
                                           int K, int k, float* v) {
  constexpr int S0 = Dims<D>::S0;
  constexpr bool BIN = std::is_same<TX, uint8_t>::value;
#pragma unroll
  for (int f = 0; f < S0; ++f) {
    float a = ld(xb + lf[f]);
    float b = ld(xb + lf[S0 + f]);
    const float* wp = w + ((long)f * K + k) * 4;
    if (BIN) {
      v[f] = (a != 0.f) ? ((b != 0.f) ? wp[3] : wp[2]) : ((b != 0.f) ? wp[1] : wp[0]);
    } else {
      v[f] = lut(a, b, wp[0], wp[1], wp[2], wp[3]);
    }
  }
#pragma unroll
  for (int lvl = 1; lvl < D; ++lvl) {
    const int n = S0 >> lvl;
    const float* wl = w + level_off(S0, lvl, K);
#pragma unroll
    for (int j = 0; j < n; ++j) {
      const float* wp = wl + ((long)j * K + k) * 4;
      v[j] = lut(v[2 * j], v[2 * j + 1], wp[0], wp[1], wp[2], wp[3]);
    }
  }
  return v[0];
}

template <typename TX, typename TY, int D>
__global__ void fwd_kernel(const TX* __restrict__ x, const int* __restrict__ leaf,
                           const int* __restrict__ pos, const float* __restrict__ w,
                           TY* __restrict__ y, uint8_t* __restrict__ arg,
                           int B, int K, int P, int Q, int Nsrc, int WIN, bool store_arg) {
  constexpr int S0 = Dims<D>::S0;
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long)B * Q) return;
  const int b = (int)(i / Q), q = (int)(i % Q);
  float v[S0];
  // grid.y is capped at 65535: each block walks the kernel indices k = y, y + gridDim.y, ...
  for (int k = blockIdx.y; k < K; k += gridDim.y) {
    float best = -INFINITY;
    int barg = 0;
    for (int j = 0; j < WIN; ++j) {
      const int p = pos[(long)q * WIN + j];
      if (p < 0) continue;
      const int* lf = leaf + ((long)k * P + p) * 2 * S0;
      const float r = tree_root<TX, D>(x + (long)b * Nsrc, lf, w, K, k, v);
      if (r > best || isnan(r)) { best = r; barg = j; }   // as max_pool2d: NaN propagates
    }
    const long o = ((long)b * K + k) * Q + q;
    st(y + o, best);
    if (store_arg) arg[o] = (uint8_t)barg;
  }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
  return v;
}

// grad kernel: all lanes take part in the shuffles (inactive lanes contribute zero).
template <typename TX, typename TG, int D>
__global__ void bwd_kernel(const TG* __restrict__ gy, const TX* __restrict__ x,
                           const int* __restrict__ leaf, const int* __restrict__ pos,
                           const uint8_t* __restrict__ arg, const float* __restrict__ w,
                           float* __restrict__ gx, float* __restrict__ gw,
                           int B, int K, int P, int Q, int Nsrc, int WIN, bool need_gx) {
  constexpr int S0 = Dims<D>::S0;
  constexpr bool BIN = std::is_same<TX, uint8_t>::value;
  // every lane of a warp runs the same k iterations, so the shuffles below stay uniform
  for (int k = blockIdx.y; k < K; k += gridDim.y) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const bool valid = i < (long)B * Q;
  const int bi = valid ? (int)(i / Q) : 0, q = valid ? (int)(i % Q) : 0;
  const long o = ((long)bi * K + k) * Q + q;
  int j = 0;
  if (valid && WIN > 1) j = arg[o];
  const int p = valid ? pos[(long)q * WIN + j] : -1;
  const bool ok = valid && p >= 0;
  float g0 = ok ? ld(gy + o) : 0.f;
  const int* lf = leaf + ((long)k * P + (ok ? p : 0)) * 2 * S0;
  const TX* xb = x + (long)bi * Nsrc;
  float la[S0], lb[S0];
#pragma unroll
  for (int f = 0; f < S0; ++f) {
    la[f] = ok ? ld(xb + lf[f]) : 0.f;
    lb[f] = ok ? ld(xb + lf[S0 + f]) : 0.f;
  }
  // forward values of every level: nodes[lvl] has S0 >> lvl entries
  float nodes[D][S0];
#pragma unroll
  for (int f = 0; f < S0; ++f) {
    const float* wp = w + ((long)f * K + k) * 4;
    if (BIN) {
      nodes[0][f] = (la[f] != 0.f) ? ((lb[f] != 0.f) ? wp[3] : wp[2])
                                   : ((lb[f] != 0.f) ? wp[1] : wp[0]);
    } else {
      nodes[0][f] = lut(la[f], lb[f], wp[0], wp[1], wp[2], wp[3]);
    }
  }
#pragma unroll
  for (int lvl = 1; lvl < D; ++lvl) {
    const int n = S0 >> lvl;
    const float* wl = w + level_off(S0, lvl, K);
#pragma unroll
    for (int jj = 0; jj < n; ++jj) {
      const float* wp = wl + ((long)jj * K + k) * 4;
      nodes[lvl][jj] = lut(nodes[lvl - 1][2 * jj], nodes[lvl - 1][2 * jj + 1],
                           wp[0], wp[1], wp[2], wp[3]);
    }
  }
  // top-down
  float g[S0];
  g[0] = g0;
#pragma unroll
  for (int lvl = D - 1; lvl >= 0; --lvl) {
    const int n = S0 >> lvl;
    const float* wl = w + level_off(S0, lvl, K);
    float gn[S0];
#pragma unroll
    for (int jj = 0; jj < n; ++jj) {
      float a, b;
      if (lvl == 0) { a = la[jj]; b = lb[jj]; }
      else { a = nodes[lvl - 1][2 * jj]; b = nodes[lvl - 1][2 * jj + 1]; }
      const float* wp = wl + ((long)jj * K + k) * 4;
      const float gg = g[jj];
      float c0, c1, c2, c3;
      if (BIN && lvl == 0) {
        const bool pa = a != 0.f, pb = b != 0.f;
        c0 = (!pa && !pb) ? gg : 0.f; c1 = (!pa && pb) ? gg : 0.f;
        c2 = (pa && !pb) ? gg : 0.f;  c3 = (pa && pb) ? gg : 0.f;
      } else {
        c0 = gg * ((1.0f - a) * (1.0f - b)); c1 = gg * ((1.0f - a) * b);
        c2 = gg * (a * (1.0f - b));          c3 = gg * (a * b);
      }
      c0 = warp_sum(c0); c1 = warp_sum(c1); c2 = warp_sum(c2); c3 = warp_sum(c3);
      if ((threadIdx.x & 31) == 0) {
        float* gp = gw + level_off(S0, lvl, K) + ((long)jj * K + k) * 4;
        atomicAdd(gp + 0, c0); atomicAdd(gp + 1, c1);
        atomicAdd(gp + 2, c2); atomicAdd(gp + 3, c3);
      }
      if (!(BIN && lvl == 0)) {
        const float ga = gg * ((wp[2] - wp[0]) * (1.0f - b) + (wp[3] - wp[1]) * b);
        const float gb = gg * ((wp[1] - wp[0]) * (1.0f - a) + (wp[3] - wp[2]) * a);
        if (lvl > 0) { gn[2 * jj] = ga; gn[2 * jj + 1] = gb; }
        else if (need_gx && ok) {
          atomicAdd(gx + (long)bi * Nsrc + lf[jj], ga);
          atomicAdd(gx + (long)bi * Nsrc + lf[S0 + jj], gb);
        }
      }
    }
    if (lvl > 0) {
#pragma unroll
      for (int jj = 0; jj < 2 * n; ++jj) g[jj] = gn[jj];
    }
  }
  }
}

#define LAUNCH_FWD(TX, TY, D)                                                              \
  fwd_kernel<TX, TY, D><<<grid, block, 0, stream>>>(                                       \
      (const TX*)x.data_ptr(), leaf.data_ptr<int>(), pos.data_ptr<int>(),                  \
      w.data_ptr<float>(), (TY*)y.data_ptr(),                                              \
      store_arg ? arg.data_ptr<uint8_t>() : nullptr, B, K, P, Q, Nsrc, WIN, store_arg)

#define SWITCH_D(TX, TY)                                                                   \
  switch (depth) {                                                                         \
    case 1: LAUNCH_FWD(TX, TY, 1); break; case 2: LAUNCH_FWD(TX, TY, 2); break;            \
    case 3: LAUNCH_FWD(TX, TY, 3); break; case 4: LAUNCH_FWD(TX, TY, 4); break;            \
    default: TORCH_CHECK(false, "depth must be 1..4");                                     \
  }

std::vector<torch::Tensor> fwd(torch::Tensor x, torch::Tensor leaf, torch::Tensor pos,
                               torch::Tensor w, int64_t depth, bool out_bf16, bool store_arg) {
  const at::cuda::CUDAGuard guard(x.device());
  const int B = x.size(0), Nsrc = x.size(1), K = leaf.size(0), P = leaf.size(1);
  const int Q = pos.size(0), WIN = pos.size(1);
  auto y = torch::empty({B, K, Q}, x.options().dtype(out_bf16 ? torch::kBFloat16 : torch::kFloat32));
  torch::Tensor arg = store_arg ? torch::empty({B, K, Q}, x.options().dtype(torch::kUInt8))
                                : torch::empty({0}, x.options().dtype(torch::kUInt8));
  const int threads = 128;
  dim3 block(threads), grid((unsigned)(((long)B * Q + threads - 1) / threads),
            (unsigned)std::min<long>(K, 65535));
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto xt = x.scalar_type();
  if (xt == torch::kFloat32) {
    if (out_bf16) { SWITCH_D(float, __nv_bfloat16) } else { SWITCH_D(float, float) }
  } else if (xt == torch::kUInt8) {
    if (out_bf16) { SWITCH_D(uint8_t, __nv_bfloat16) } else { SWITCH_D(uint8_t, float) }
  } else if (xt == torch::kBFloat16) {
    if (out_bf16) { SWITCH_D(__nv_bfloat16, __nv_bfloat16) } else { SWITCH_D(__nv_bfloat16, float) }
  } else { TORCH_CHECK(false, "x must be float32, uint8 or bfloat16"); }
  return {y, arg};
}

#define LAUNCH_BWD(TX, TG, D)                                                              \
  bwd_kernel<TX, TG, D><<<grid, block, 0, stream>>>(                                       \
      (const TG*)gy.data_ptr(), (const TX*)x.data_ptr(), leaf.data_ptr<int>(),             \
      pos.data_ptr<int>(), WIN > 1 ? arg.data_ptr<uint8_t>() : nullptr,                    \
      w.data_ptr<float>(), need_gx ? gx.data_ptr<float>() : nullptr, gw.data_ptr<float>(), \
      B, K, P, Q, Nsrc, WIN, need_gx)

#define SWITCH_DB(TX, TG)                                                                  \
  switch (depth) {                                                                         \
    case 1: LAUNCH_BWD(TX, TG, 1); break; case 2: LAUNCH_BWD(TX, TG, 2); break;            \
    case 3: LAUNCH_BWD(TX, TG, 3); break; case 4: LAUNCH_BWD(TX, TG, 4); break;            \
    default: TORCH_CHECK(false, "depth must be 1..4");                                     \
  }

std::vector<torch::Tensor> bwd(torch::Tensor gy, torch::Tensor x, torch::Tensor leaf,
                               torch::Tensor pos, torch::Tensor arg, torch::Tensor w,
                               int64_t depth, bool need_gx) {
  const at::cuda::CUDAGuard guard(x.device());
  const int B = x.size(0), Nsrc = x.size(1), K = leaf.size(0), P = leaf.size(1);
  const int Q = pos.size(0), WIN = pos.size(1);
  gy = gy.contiguous();
  auto gw = torch::zeros_like(w);
  torch::Tensor gx = need_gx ? torch::zeros({B, Nsrc}, x.options().dtype(torch::kFloat32))
                             : torch::empty({0}, x.options().dtype(torch::kFloat32));
  const int threads = 128;
  dim3 block(threads), grid((unsigned)(((long)B * Q + threads - 1) / threads),
            (unsigned)std::min<long>(K, 65535));
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto xt = x.scalar_type();
  const bool g_bf16 = gy.scalar_type() == torch::kBFloat16;
  if (xt == torch::kFloat32) {
    if (g_bf16) { SWITCH_DB(float, __nv_bfloat16) } else { SWITCH_DB(float, float) }
  } else if (xt == torch::kUInt8) {
    if (g_bf16) { SWITCH_DB(uint8_t, __nv_bfloat16) } else { SWITCH_DB(uint8_t, float) }
  } else if (xt == torch::kBFloat16) {
    if (g_bf16) { SWITCH_DB(__nv_bfloat16, __nv_bfloat16) } else { SWITCH_DB(__nv_bfloat16, float) }
  } else { TORCH_CHECK(false, "x must be float32, uint8 or bfloat16"); }
  return {gx, gw};
}
"""

_mod = None


def _load():
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load_inline
        _mod = load_inline(
            name="torchlogix_tree_cuda", cpp_sources=_CPP, cuda_sources=_CUDA,
            functions=["fwd", "bwd"], extra_cuda_cflags=["-O3", "--fmad=false"],
            verbose=False)
    return _mod


def fwd(x, leaf, pos, w, depth, binary, out_dtype, need_arg):
    import torch
    store_arg = bool(need_arg and pos.shape[1] > 1)
    y, arg = _load().fwd(x.contiguous(), leaf, pos, w, depth,
                         out_dtype == torch.bfloat16, store_arg)
    return y, arg


def bwd(gy, x, leaf, pos, arg, w, depth, binary, need_gx):
    gx, gw = _load().bwd(gy, x.contiguous(), leaf, pos, arg, w, depth,
                         bool(need_gx and not binary))
    return (gx if need_gx and not binary else None), gw
