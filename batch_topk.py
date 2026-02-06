"""
Batch Top-K — single-file, JIT-compiled.

Usage:
    from batch_topk import batch_topk, get_buffer

    batch_topk(metric, topks, valid_lens, out_idxs)

Input:  metric (batch_size, max_len) — float16/bfloat16/float32
Output: out_idxs (batch_size, max_k) — int32, filled in-place
"""

import os
import torch
from typing import Optional
from torch.utils.cpp_extension import load_inline

# ----------------------------------------------------------------
# CUDA kernel source
# ----------------------------------------------------------------

_CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cub/block/block_scan.cuh>
#include <cub/block/radix_rank_sort_operations.cuh>

namespace batch_topk {

// ============================================================================
// Configuration
// ============================================================================

constexpr int BitsPerPass = 8;
constexpr int BlockSize = 512;
constexpr int NumBuckets = 1 << BitsPerPass;

// Adaptive algorithm thresholds (empirically determined)
constexpr int SEQLEN_THRESHOLD = 32768;
constexpr int BATCH_SEQLEN_RATIO = 2048;
constexpr int MIN_ELEMENTS_PER_BLOCK = 4096;

// ============================================================================
// Internal Helpers
// ============================================================================

namespace detail {

template <typename T>
__host__ __device__ constexpr T upper_bound() {
    if constexpr (std::is_same_v<T, __half>) {
        return __ushort_as_half(0x7C00u);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __ushort_as_bfloat16(0x7F80u);
    } else {
        return std::numeric_limits<T>::max();
    }
}

template <typename T>
__host__ __device__ constexpr T lower_bound() {
    if constexpr (std::is_same_v<T, __half>) {
        return __ushort_as_half(0xFC00u);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __ushort_as_bfloat16(0xFF80u);
    } else {
        return std::numeric_limits<T>::lowest();
    }
}

template <typename T>
__host__ __device__ constexpr int calc_num_passes() {
    return (sizeof(T) * 8 + BitsPerPass - 1) / BitsPerPass;
}

template <typename T>
__device__ constexpr int calc_start_bit(int pass) {
    int start_bit = static_cast<int>(sizeof(T) * 8) - (pass + 1) * BitsPerPass;
    return start_bit < 0 ? 0 : start_bit;
}

template <typename T>
__device__ constexpr unsigned calc_mask(int pass) {
    int num_bits = calc_start_bit<T>(pass - 1) - calc_start_bit<T>(pass);
    return (1 << num_bits) - 1;
}

template <typename T>
__device__ typename cub::Traits<T>::UnsignedBits twiddle_in(T key, bool select_min) {
    auto bits = reinterpret_cast<typename cub::Traits<T>::UnsignedBits&>(key);
    bits = cub::Traits<T>::TwiddleIn(bits);
    if (!select_min) bits = ~bits;
    return bits;
}

template <typename T>
__device__ int calc_bucket(T x, int start_bit, unsigned mask, bool select_min) {
    return (twiddle_in(x, select_min) >> start_bit) & mask;
}

inline bool should_use_multiblock(size_t batch_size, size_t max_len) {
    if (max_len <= SEQLEN_THRESHOLD) return false;
    return batch_size < (max_len / BATCH_SEQLEN_RATIO);
}

// ============================================================================
// Single-Block Kernel (one block per row)
// ============================================================================

template <typename T>
struct alignas(128) Counter {
    int32_t k;
    int32_t len;
    int32_t previous_len;
    typename cub::Traits<T>::UnsignedBits kth_value_bits;
    alignas(128) int32_t out_cnt;
    alignas(128) int32_t out_back_cnt;
};

template <typename T>
__device__ void filter_and_histogram(
    const T* in_buf, int32_t previous_len,
    Counter<T>* counter, int32_t* histogram,
    bool select_min, int pass) {

    for (int i = threadIdx.x; i < NumBuckets; i += blockDim.x) {
        histogram[i] = 0;
    }
    __syncthreads();

    const int start_bit = calc_start_bit<T>(pass);
    const unsigned mask = calc_mask<T>(pass);

    if (pass == 0) {
        for (int32_t i = threadIdx.x; i < previous_len; i += blockDim.x) {
            int bucket = calc_bucket(in_buf[i], start_bit, mask, select_min);
            atomicAdd(&histogram[bucket], 1);
        }
    } else {
        const auto kth_value_bits = counter->kth_value_bits;
        const int prev_start_bit = calc_start_bit<T>(pass - 1);

        for (int32_t i = threadIdx.x; i < previous_len; i += blockDim.x) {
            const T value = in_buf[i];
            const auto prev_bits = (twiddle_in(value, select_min) >> prev_start_bit) << prev_start_bit;
            if (prev_bits == kth_value_bits) {
                int bucket = calc_bucket(value, start_bit, mask, select_min);
                atomicAdd(&histogram[bucket], 1);
            }
        }
    }
}

template <typename T>
__device__ void choose_bucket(Counter<T>* counter, int32_t* histogram, int32_t k, int pass) {
    __shared__ int32_t scan[NumBuckets];
    if (threadIdx.x < NumBuckets) scan[threadIdx.x] = histogram[threadIdx.x];
    __syncthreads();

    if (threadIdx.x == 0) {
        int32_t sum = 0;
        for (int i = 0; i < NumBuckets; i++) {
            sum += scan[i];
            scan[i] = sum;
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < NumBuckets; i += blockDim.x) {
        int32_t prev = (i == 0) ? 0 : scan[i - 1];
        int32_t cur = scan[i];
        if (prev < k && cur >= k) {
            counter->k = k - prev;
            counter->len = cur - prev;
            int start_bit = calc_start_bit<T>(pass);
            counter->kth_value_bits |= (static_cast<typename cub::Traits<T>::UnsignedBits>(i) << start_bit);
        }
    }
}

template <typename T>
__device__ void last_filter(
    const T* in_buf, int32_t* out_idx,
    int32_t current_len, int32_t k,
    Counter<T>* counter, bool select_min, int pass) {

    const auto kth_value_bits = counter->kth_value_bits;
    const int start_bit = calc_start_bit<T>(pass);
    const int32_t num_of_kth_needed = counter->k;

    for (int32_t i = threadIdx.x; i < current_len; i += blockDim.x) {
        const T value = in_buf[i];
        const auto bits = (twiddle_in(value, select_min) >> start_bit) << start_bit;
        if (bits < kth_value_bits) {
            int32_t pos = atomicAdd(&counter->out_cnt, 1);
            out_idx[pos] = i;
        } else if (bits == kth_value_bits) {
            int32_t back_pos = atomicAdd(&counter->out_back_cnt, 1);
            if (back_pos < num_of_kth_needed) {
                out_idx[k - 1 - back_pos] = i;
            }
        }
    }
}

template <typename T>
__global__ void SingleBlockKernel(
    const T* __restrict__ in_val,
    const int32_t* __restrict__ valid_lens,
    const int32_t* __restrict__ ks,
    int32_t* __restrict__ out_idx,
    char* __restrict__ bufs,
    int32_t batch_size, int32_t max_len, int32_t max_k,
    bool select_min) {

    __shared__ Counter<T> counter;
    __shared__ int32_t histogram[NumBuckets];

    const int32_t batch_id = blockIdx.x;
    if (batch_id >= batch_size) return;

    const int32_t l_len = valid_lens[batch_id];
    const int32_t k = ks[batch_id];
    if (k == 0) return;

    const T* row_in = in_val + batch_id * max_len;
    int32_t* row_out_idx = out_idx + batch_id * max_k;

    if (l_len <= k) {
        for (int32_t i = threadIdx.x; i < l_len; i += blockDim.x) {
            row_out_idx[i] = i;
        }
        return;
    }

    int32_t buf_len = max_k / 8;
    buf_len = (buf_len / 64) * 64;
    buf_len = buf_len > 256 ? buf_len : 256;
    int32_t* row_buf = reinterpret_cast<int32_t*>(bufs + batch_id * buf_len * sizeof(int32_t));

    if (threadIdx.x == 0) {
        counter.k = k;
        counter.len = l_len;
        counter.previous_len = l_len;
        counter.kth_value_bits = 0;
        counter.out_cnt = 0;
        counter.out_back_cnt = 0;
    }
    __syncthreads();

    constexpr int num_passes = calc_num_passes<T>();

    for (int pass = 0; pass < num_passes; ++pass) {
        const int32_t current_len = counter.len;
        const int32_t current_k = counter.k;

        filter_and_histogram(row_in, counter.previous_len,
                            &counter, histogram, select_min, pass);
        __syncthreads();

        choose_bucket(&counter, histogram, current_k, pass);
        if (threadIdx.x == 0) counter.previous_len = current_len;
        __syncthreads();

        if (counter.len == counter.k || pass == num_passes - 1) {
            last_filter(row_in, row_out_idx, l_len, k,
                       &counter, select_min, pass);
            break;
        }
    }
}

// ============================================================================
// Multi-Block Kernels (multiple blocks per row)
// ============================================================================

template <typename T>
__global__ void MultiBlockHistogramKernel(
    const T* __restrict__ in_val,
    const int32_t* __restrict__ valid_lens,
    int32_t* __restrict__ histograms,
    const uint32_t* __restrict__ kth_bits,
    int32_t batch_size, int32_t max_len,
    int blocks_per_row, int pass, bool select_min) {

    const int32_t batch_id = blockIdx.x / blocks_per_row;
    const int block_in_row = blockIdx.x % blocks_per_row;
    if (batch_id >= batch_size) return;

    const int32_t l_len = valid_lens[batch_id];
    const T* row_in = in_val + batch_id * max_len;
    int32_t* row_hist = histograms + batch_id * NumBuckets;

    int32_t elems_per_block = (l_len + blocks_per_row - 1) / blocks_per_row;
    int32_t start = block_in_row * elems_per_block;
    int32_t end = min(start + elems_per_block, l_len);

    const int start_bit = calc_start_bit<T>(pass);
    const unsigned mask = calc_mask<T>(pass);

    if (pass == 0) {
        for (int32_t i = start + threadIdx.x; i < end; i += blockDim.x) {
            int bucket = calc_bucket(row_in[i], start_bit, mask, select_min);
            atomicAdd(&row_hist[bucket], 1);
        }
    } else {
        const uint32_t kth_val = kth_bits[batch_id];
        const int prev_start = calc_start_bit<T>(pass - 1);
        for (int32_t i = start + threadIdx.x; i < end; i += blockDim.x) {
            const T value = row_in[i];
            const auto bits = static_cast<uint32_t>(
                (twiddle_in(value, select_min) >> prev_start) << prev_start);
            if (bits == kth_val) {
                int bucket = calc_bucket(value, start_bit, mask, select_min);
                atomicAdd(&row_hist[bucket], 1);
            }
        }
    }
}

template <typename T>
__global__ void MultiBlockChooseBucketKernel(
    int32_t* __restrict__ histograms,
    int32_t* __restrict__ remaining_ks,
    uint32_t* __restrict__ kth_bits,
    int32_t batch_size, int pass) {

    const int32_t batch_id = blockIdx.x;
    if (batch_id >= batch_size) return;

    int32_t* hist = histograms + batch_id * NumBuckets;
    int32_t k = remaining_ks[batch_id];

    __shared__ int32_t scan[NumBuckets];
    if (threadIdx.x < NumBuckets) scan[threadIdx.x] = hist[threadIdx.x];
    __syncthreads();

    if (threadIdx.x == 0) {
        int32_t sum = 0;
        for (int i = 0; i < NumBuckets; i++) { sum += scan[i]; scan[i] = sum; }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < NumBuckets; i += blockDim.x) {
        int32_t prev = (i == 0) ? 0 : scan[i - 1];
        if (prev < k && scan[i] >= k) {
            remaining_ks[batch_id] = k - prev;
            kth_bits[batch_id] |= (static_cast<uint32_t>(i) << calc_start_bit<T>(pass));
        }
    }

    if (threadIdx.x < NumBuckets) hist[threadIdx.x] = 0;
}

template <typename T>
__global__ void MultiBlockFilterKernel(
    const T* __restrict__ in_val,
    const int32_t* __restrict__ valid_lens,
    const int32_t* __restrict__ original_ks,
    int32_t* __restrict__ out_idx,
    const int32_t* __restrict__ remaining_ks,
    const uint32_t* __restrict__ kth_bits,
    int32_t* __restrict__ out_cnts,
    int32_t* __restrict__ out_back_cnts,
    int32_t batch_size, int32_t max_len, int32_t max_k,
    int blocks_per_row, int pass, bool select_min) {

    const int32_t batch_id = blockIdx.x / blocks_per_row;
    const int block_in_row = blockIdx.x % blocks_per_row;
    if (batch_id >= batch_size) return;

    const int32_t l_len = valid_lens[batch_id];
    const int32_t k = original_ks[batch_id];
    if (k == 0 || l_len <= k) return;

    const T* row_in = in_val + batch_id * max_len;
    int32_t* row_out = out_idx + batch_id * max_k;

    int32_t elems_per_block = (l_len + blocks_per_row - 1) / blocks_per_row;
    int32_t start = block_in_row * elems_per_block;
    int32_t end = min(start + elems_per_block, l_len);

    const uint32_t kth_val = kth_bits[batch_id];
    const int start_bit = calc_start_bit<T>(pass);
    const int32_t num_kth = remaining_ks[batch_id];

    for (int32_t i = start + threadIdx.x; i < end; i += blockDim.x) {
        const auto bits = static_cast<uint32_t>(
            (twiddle_in(row_in[i], select_min) >> start_bit) << start_bit);
        if (bits < kth_val) {
            row_out[atomicAdd(&out_cnts[batch_id], 1)] = i;
        } else if (bits == kth_val) {
            int32_t back = atomicAdd(&out_back_cnts[batch_id], 1);
            if (back < num_kth) row_out[k - 1 - back] = i;
        }
    }
}

template <typename T>
__global__ void HandleTrivialKernel(
    const int32_t* __restrict__ valid_lens,
    const int32_t* __restrict__ ks,
    int32_t* __restrict__ out_idx,
    int32_t batch_size, int32_t max_k) {

    const int32_t batch_id = blockIdx.x;
    if (batch_id >= batch_size) return;

    const int32_t l_len = valid_lens[batch_id];
    const int32_t k = ks[batch_id];
    if (k == 0 || l_len > k) return;

    int32_t* row_out = out_idx + batch_id * max_k;
    for (int32_t i = threadIdx.x; i < l_len; i += blockDim.x) {
        row_out[i] = i;
    }
}

// ============================================================================
// Dispatch Functions
// ============================================================================

template <typename T>
void launch_single_block(
    const T* in_val, const int32_t* valid_lens, const int32_t* ks,
    int32_t* out_idx, char* buf,
    size_t batch_size, size_t max_len, size_t max_k,
    bool select_min, cudaStream_t stream) {

    SingleBlockKernel<T><<<batch_size, BlockSize, 0, stream>>>(
        in_val, valid_lens, ks, out_idx, buf,
        batch_size, max_len, max_k, select_min);
}

template <typename T>
void launch_multi_block(
    const T* in_val, const int32_t* valid_lens, const int32_t* ks,
    int32_t* out_idx, char* workspace, size_t workspace_size,
    size_t batch_size, size_t max_len, size_t max_k,
    bool select_min, cudaStream_t stream) {

    constexpr int num_passes = calc_num_passes<T>();

    int blocks_per_row = (max_len + MIN_ELEMENTS_PER_BLOCK - 1) / MIN_ELEMENTS_PER_BLOCK;
    blocks_per_row = max(blocks_per_row, 1);
    blocks_per_row = min(blocks_per_row, 64);

    size_t offset = 0;
    int32_t* histograms = reinterpret_cast<int32_t*>(workspace + offset);
    offset += batch_size * NumBuckets * sizeof(int32_t);
    uint32_t* kth_bits = reinterpret_cast<uint32_t*>(workspace + offset);
    offset += batch_size * sizeof(uint32_t);
    int32_t* remaining_ks = reinterpret_cast<int32_t*>(workspace + offset);
    offset += batch_size * sizeof(int32_t);
    int32_t* out_cnts = reinterpret_cast<int32_t*>(workspace + offset);
    offset += batch_size * sizeof(int32_t);
    int32_t* out_back_cnts = reinterpret_cast<int32_t*>(workspace + offset);

    cudaMemsetAsync(workspace, 0, workspace_size, stream);
    cudaMemcpyAsync(remaining_ks, ks, batch_size * sizeof(int32_t),
                    cudaMemcpyDeviceToDevice, stream);

    HandleTrivialKernel<T><<<batch_size, BlockSize, 0, stream>>>(
        valid_lens, ks, out_idx, batch_size, max_k);

    for (int pass = 0; pass < num_passes; ++pass) {
        MultiBlockHistogramKernel<T><<<batch_size * blocks_per_row, BlockSize, 0, stream>>>(
            in_val, valid_lens, histograms, kth_bits,
            batch_size, max_len, blocks_per_row, pass, select_min);

        MultiBlockChooseBucketKernel<T><<<batch_size, BlockSize, 0, stream>>>(
            histograms, remaining_ks, kth_bits, batch_size, pass);
    }

    MultiBlockFilterKernel<T><<<batch_size * blocks_per_row, BlockSize, 0, stream>>>(
        in_val, valid_lens, ks, out_idx, remaining_ks, kth_bits,
        out_cnts, out_back_cnts, batch_size, max_len, max_k,
        blocks_per_row, num_passes - 1, select_min);
}

} // namespace detail

// ============================================================================
// Type Dispatch
// ============================================================================

#define DISPATCH_FLOAT_TYPES(dtype, DType, ...)                               \
    [&]() {                                                                    \
        if (dtype == at::ScalarType::Half) {                                   \
            using DType = __half; return __VA_ARGS__();                        \
        } else if (dtype == at::ScalarType::BFloat16) {                        \
            using DType = __nv_bfloat16; return __VA_ARGS__();                 \
        } else if (dtype == at::ScalarType::Float) {                           \
            using DType = float; return __VA_ARGS__();                         \
        } else {                                                               \
            TORCH_CHECK(false, "Unsupported dtype"); return false;             \
        }                                                                      \
    }()

} // namespace batch_topk

// ============================================================================
// Functions exposed to Python via load_inline
// ============================================================================

int64_t get_buffer_size(int64_t batch_size, int64_t max_len, int64_t max_k) {
    int64_t buf_len = max_k / 8;
    buf_len = (buf_len / 64) * 64;
    buf_len = buf_len > 256 ? buf_len : 256;
    int64_t single_block_size = batch_size * buf_len * sizeof(int32_t);
    int64_t multi_block_size = batch_size * (batch_topk::NumBuckets + 5) * sizeof(int32_t) + 256;
    return std::max(single_block_size, multi_block_size);
}

void launch_batch_topk(
    at::Tensor metric,
    at::Tensor topks,
    at::Tensor valid_lens,
    at::Tensor out_idxs,
    at::Tensor buf,
    bool select_min) {

    TORCH_CHECK(metric.dim() == 2, "metric must be 2D");
    TORCH_CHECK(out_idxs.dim() == 2, "out_idxs must be 2D");
    TORCH_CHECK(metric.is_cuda() && out_idxs.is_cuda(), "tensors must be on CUDA");
    TORCH_CHECK(metric.is_contiguous() && out_idxs.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(out_idxs.scalar_type() == at::ScalarType::Int, "out_idxs must be int32");

    const size_t batch_size = metric.size(0);
    const size_t max_len = metric.size(1);
    const size_t max_k = out_idxs.size(1);

    if (topks.scalar_type() != at::ScalarType::Int)
        topks = topks.to(at::ScalarType::Int);
    if (valid_lens.scalar_type() != at::ScalarType::Int)
        valid_lens = valid_lens.to(at::ScalarType::Int);

    const c10::cuda::OptionalCUDAGuard guard(metric.device());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const size_t buf_size = buf.numel() * buf.element_size();

    bool use_multiblock = batch_topk::detail::should_use_multiblock(batch_size, max_len);

    DISPATCH_FLOAT_TYPES(metric.scalar_type(), DType, [&] {
        if (use_multiblock) {
            batch_topk::detail::launch_multi_block<DType>(
                static_cast<DType*>(metric.data_ptr()),
                static_cast<int32_t*>(valid_lens.data_ptr()),
                static_cast<int32_t*>(topks.data_ptr()),
                static_cast<int32_t*>(out_idxs.data_ptr()),
                static_cast<char*>(buf.data_ptr()),
                buf_size, batch_size, max_len, max_k, select_min, stream);
        } else {
            batch_topk::detail::launch_single_block<DType>(
                static_cast<DType*>(metric.data_ptr()),
                static_cast<int32_t*>(valid_lens.data_ptr()),
                static_cast<int32_t*>(topks.data_ptr()),
                static_cast<int32_t*>(out_idxs.data_ptr()),
                static_cast<char*>(buf.data_ptr()),
                batch_size, max_len, max_k, select_min, stream);
        }
        return true;
    });
}
"""

_CPP_SRC = """
int64_t get_buffer_size(int64_t batch_size, int64_t max_len, int64_t max_k);
void launch_batch_topk(
    at::Tensor metric,
    at::Tensor topks,
    at::Tensor valid_lens,
    at::Tensor out_idxs,
    at::Tensor buf,
    bool select_min);
"""

# ----------------------------------------------------------------
# JIT compile (cached after first call)
# ----------------------------------------------------------------

_module = None


def _get_module():
    global _module
    if _module is not None:
        return _module

    os.environ.setdefault(
        "TORCH_CUDA_ARCH_LIST",
        "8.0;8.9;9.0",
    )

    _module = load_inline(
        name="batch_topk_jit",
        cpp_sources=_CPP_SRC,
        cuda_sources=_CUDA_SRC,
        functions=["launch_batch_topk", "get_buffer_size"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
        ],
        verbose=False,
    )
    return _module


# ----------------------------------------------------------------
# Public API
# ----------------------------------------------------------------

def get_buffer(
    batch_size: int,
    max_len: int,
    max_k: int,
    device: torch.device,
) -> torch.Tensor:
    """Allocate workspace buffer for batch_topk.

    Args:
        batch_size: Number of rows in the batch.
        max_len:    Maximum sequence length.
        max_k:      Maximum k value across all rows.
        device:     CUDA device.

    Returns:
        Workspace buffer tensor.
    """
    buf_size = _get_module().get_buffer_size(batch_size, max_len, max_k)
    return torch.empty(buf_size, dtype=torch.uint8, device=device)


def batch_topk(
    metric: torch.Tensor,
    topks: torch.Tensor,
    valid_lens: torch.Tensor,
    out_idxs: torch.Tensor,
    buf: Optional[torch.Tensor] = None,
    select_min: bool = False,
) -> None:
    """Batch top-k selection with variable k per row.

    Selects top-k elements from each row, filling out_idxs with indices.
    Automatically selects best algorithm based on batch size and seqlen.

    Args:
        metric:     Input tensor of shape (batch_size, max_len).
        topks:      K values for each row, shape (batch_size,).
        valid_lens: Valid length for each row, shape (batch_size,).
        out_idxs:   Output indices, shape (batch_size, max_k), filled in-place.
        buf:        Optional pre-allocated workspace buffer.
        select_min: If True, select minimum values; otherwise maximum.
    """
    assert metric.is_cuda, "metric must be on CUDA"
    assert metric.dim() == 2, "metric must be 2D"
    assert metric.is_contiguous(), "metric must be contiguous"
    assert out_idxs.dim() == 2, "out_idxs must be 2D"
    assert out_idxs.dtype == torch.int32, "out_idxs must be int32"
    assert out_idxs.is_contiguous(), "out_idxs must be contiguous"
    assert topks.dtype == torch.int32, "topks must be int32"
    assert topks.is_contiguous(), "topks must be contiguous"
    assert valid_lens.dtype == torch.int32, "valid_lens must be int32"
    assert valid_lens.is_contiguous(), "valid_lens must be contiguous"

    batch_size = metric.size(0)
    max_len = metric.size(1)
    max_k = out_idxs.size(1)

    if buf is None:
        buf = get_buffer(batch_size, max_len, max_k, metric.device)

    _get_module().launch_batch_topk(
        metric, topks, valid_lens, out_idxs, buf, select_min
    )


__all__ = ["batch_topk", "get_buffer"]
