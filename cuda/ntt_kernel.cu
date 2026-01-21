#include <cuda_runtime.h>
#include <stdint.h>

// BabyBear prime: 2^31 - 2^27 + 1
#define BABYBEAR_PRIME 2013265921ULL

// BabyBear field element
typedef uint64_t BabyBearElem;

// Device and host functions for BabyBear field arithmetic
__device__ __host__ __forceinline__ uint64_t bb_reduce(uint64_t val) {
    if (val >= BABYBEAR_PRIME) {
        val -= BABYBEAR_PRIME;
    }
    if (val >= BABYBEAR_PRIME) {
        val -= BABYBEAR_PRIME;
    }
    return val;
}

__device__ __host__ __forceinline__ uint64_t bb_add(uint64_t a, uint64_t b) {
    uint64_t sum = a + b;
    return bb_reduce(sum);
}

__device__ __host__ __forceinline__ uint64_t bb_sub(uint64_t a, uint64_t b) {
    if (a >= b) {
        return a - b;
    } else {
        return a + BABYBEAR_PRIME - b;
    }
}

__device__ __host__ __forceinline__ uint64_t bb_mul(uint64_t a, uint64_t b) {
    // Use 128-bit intermediate to avoid overflow
#ifdef __CUDA_ARCH__
    // Device code: use CUDA intrinsic
    unsigned long long product_hi, product_lo;
    product_lo = a * b;
    product_hi = __umul64hi(a, b);
    uint64_t result = product_lo % BABYBEAR_PRIME;
    return result;
#else
    // Host code: use standard C++
    __uint128_t product = (__uint128_t)a * (__uint128_t)b;
    uint64_t result = product % BABYBEAR_PRIME;
    return result;
#endif
}

// Modular exponentiation on device and host
__device__ __host__ uint64_t bb_pow(uint64_t base, uint64_t exp) {
    uint64_t result = 1;
    while (exp > 0) {
        if (exp & 1) {
            result = bb_mul(result, base);
        }
        base = bb_mul(base, base);
        exp >>= 1;
    }
    return result;
}

// Note: Primitive 2^27-th root of unity for BabyBear is 440564289 (31^15 mod p)

// Bit reversal
__device__ uint32_t bit_reverse(uint32_t x, uint32_t log_n) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < log_n; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

// Cooley-Tukey NTT kernel
// Each block handles a segment of the array
__global__ void ntt_kernel_bit_reverse(uint64_t* values, uint32_t n, uint32_t log_n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        uint32_t rev_idx = bit_reverse(idx, log_n);
        if (idx < rev_idx) {
            uint64_t temp = values[idx];
            values[idx] = values[rev_idx];
            values[rev_idx] = temp;
        }
    }
}

// Butterfly operation kernel for a specific stage
__global__ void ntt_kernel_butterfly(uint64_t* values, uint32_t n, uint32_t stage, uint64_t omega) {
    uint32_t len = 1 << stage; // 2^stage
    uint32_t half_len = len >> 1;

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total_butterflies = n / 2;

    if (idx < total_butterflies) {
        uint32_t group = idx / half_len;
        uint32_t pos_in_group = idx % half_len;
        uint32_t base = group * len;

        uint32_t i = base + pos_in_group;
        uint32_t j = i + half_len;

        // Compute twiddle factor: w = omega^(pos_in_group * (n / len))
        uint32_t step = n / len;
        uint64_t twiddle_exp = (uint64_t)pos_in_group * step;
        uint64_t w = bb_pow(omega, twiddle_exp);

        uint64_t u = values[i];
        uint64_t v = bb_mul(values[j], w);

        values[i] = bb_add(u, v);
        values[j] = bb_sub(u, v);
    }
}

// Optimized butterfly kernel using shared memory
__global__ void ntt_kernel_butterfly_shared(uint64_t* values, uint32_t n, uint32_t stage, uint64_t omega) {
    extern __shared__ uint64_t shared_data[];

    uint32_t len = 1 << stage;
    uint32_t half_len = len >> 1;

    uint32_t tid = threadIdx.x;
    uint32_t block_start = blockIdx.x * blockDim.x * 2;

    // Load data into shared memory
    if (block_start + tid < n) {
        shared_data[tid] = values[block_start + tid];
    }
    if (block_start + blockDim.x + tid < n) {
        shared_data[blockDim.x + tid] = values[block_start + blockDim.x + tid];
    }
    __syncthreads();

    // Perform butterfly operations
    if (tid < half_len && block_start + tid < n) {
        uint32_t step = n / len;
        uint64_t twiddle_exp = (uint64_t)tid * step;
        uint64_t w = bb_pow(omega, twiddle_exp);

        uint64_t u = shared_data[tid];
        uint64_t v = bb_mul(shared_data[tid + half_len], w);

        shared_data[tid] = bb_add(u, v);
        shared_data[tid + half_len] = bb_sub(u, v);
    }
    __syncthreads();

    // Write back to global memory
    if (block_start + tid < n) {
        values[block_start + tid] = shared_data[tid];
    }
    if (block_start + blockDim.x + tid < n) {
        values[block_start + blockDim.x + tid] = shared_data[blockDim.x + tid];
    }
}

// Scale by inverse N for INTT
__global__ void scale_by_inv_n(uint64_t* values, uint32_t n, uint64_t inv_n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        values[idx] = bb_mul(values[idx], inv_n);
    }
}

// C interface functions
extern "C" {

// Forward NTT
void cuda_ntt(uint64_t* d_values, uint32_t n, uint64_t omega) {
    uint32_t log_n = 0;
    uint32_t temp = n;
    while (temp > 1) {
        log_n++;
        temp >>= 1;
    }

    // Bit reversal
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    ntt_kernel_bit_reverse<<<blocks, threads>>>(d_values, n, log_n);
    cudaDeviceSynchronize();

    // Butterfly stages
    for (uint32_t stage = 1; stage <= log_n; stage++) {
        uint32_t total_butterflies = n / 2;
        blocks = (total_butterflies + threads - 1) / threads;
        ntt_kernel_butterfly<<<blocks, threads>>>(d_values, n, stage, omega);
        cudaDeviceSynchronize();
    }
}

// Inverse NTT
void cuda_intt(uint64_t* d_values, uint32_t n, uint64_t omega) {
    // Compute omega^-1 = omega^(n-1)
    uint64_t inv_omega = bb_pow(omega, n - 1);

    // Perform NTT with inverse omega
    cuda_ntt(d_values, n, inv_omega);

    // Scale by 1/n
    uint64_t inv_n = bb_pow(n, BABYBEAR_PRIME - 2); // n^-1 mod p
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    scale_by_inv_n<<<blocks, threads>>>(d_values, n, inv_n);
    cudaDeviceSynchronize();
}

// Helper to copy data to device
cudaError_t cuda_copy_to_device(uint64_t* d_dest, const uint64_t* h_src, size_t count) {
    return cudaMemcpy(d_dest, h_src, count * sizeof(uint64_t), cudaMemcpyHostToDevice);
}

// Helper to copy data from device
cudaError_t cuda_copy_from_device(uint64_t* h_dest, const uint64_t* d_src, size_t count) {
    return cudaMemcpy(h_dest, d_src, count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
}

// Allocate device memory
cudaError_t cuda_malloc(uint64_t** d_ptr, size_t count) {
    return cudaMalloc((void**)d_ptr, count * sizeof(uint64_t));
}

// Free device memory
cudaError_t cuda_free(uint64_t* d_ptr) {
    return cudaFree(d_ptr);
}

// Get CUDA error string
const char* cuda_get_error_string(cudaError_t error) {
    return cudaGetErrorString(error);
}

// ============================================================================
// BATCHED NTT KERNELS - Process many NTTs in parallel for maximum GPU throughput
// ============================================================================

// Batched bit reversal: process multiple NTTs at once
// Each NTT is at offset (batch_idx * stride) in the array
__global__ void ntt_batched_bit_reverse(uint64_t* values, uint32_t num_ntts, uint32_t ntt_size, uint32_t stride, uint32_t log_n) {
    uint32_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total_elements = num_ntts * ntt_size;

    if (global_idx >= total_elements) return;

    uint32_t batch_idx = global_idx / ntt_size;
    uint32_t local_idx = global_idx % ntt_size;
    uint32_t rev_idx = bit_reverse(local_idx, log_n);

    if (local_idx < rev_idx) {
        uint64_t* ntt_base = values + batch_idx * stride;
        uint64_t temp = ntt_base[local_idx];
        ntt_base[local_idx] = ntt_base[rev_idx];
        ntt_base[rev_idx] = temp;
    }
}

// Batched butterfly: process all butterflies for a stage across all NTTs
__global__ void ntt_batched_butterfly(uint64_t* values, uint32_t num_ntts, uint32_t ntt_size, uint32_t stride, uint32_t stage, uint64_t omega) {
    uint32_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t butterflies_per_ntt = ntt_size / 2;
    uint32_t total_butterflies = num_ntts * butterflies_per_ntt;

    if (global_idx >= total_butterflies) return;

    uint32_t batch_idx = global_idx / butterflies_per_ntt;
    uint32_t local_butterfly = global_idx % butterflies_per_ntt;

    uint32_t len = 1 << stage;
    uint32_t half_len = len >> 1;

    uint32_t group = local_butterfly / half_len;
    uint32_t pos_in_group = local_butterfly % half_len;
    uint32_t base = group * len;

    uint32_t i = base + pos_in_group;
    uint32_t j = i + half_len;

    uint64_t* ntt_base = values + batch_idx * stride;

    // Compute twiddle factor
    uint32_t step = ntt_size / len;
    uint64_t twiddle_exp = (uint64_t)pos_in_group * step;
    uint64_t w = bb_pow(omega, twiddle_exp);

    uint64_t u = ntt_base[i];
    uint64_t v = bb_mul(ntt_base[j], w);

    ntt_base[i] = bb_add(u, v);
    ntt_base[j] = bb_sub(u, v);
}

// Batched scale by inverse N
__global__ void ntt_batched_scale(uint64_t* values, uint32_t num_ntts, uint32_t ntt_size, uint32_t stride, uint64_t inv_n) {
    uint32_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total_elements = num_ntts * ntt_size;

    if (global_idx >= total_elements) return;

    uint32_t batch_idx = global_idx / ntt_size;
    uint32_t local_idx = global_idx % ntt_size;

    uint64_t* ntt_base = values + batch_idx * stride;
    ntt_base[local_idx] = bb_mul(ntt_base[local_idx], inv_n);
}

// Batched NTT: process num_ntts NTTs in parallel
// values: array of NTTs, each at offset (i * stride)
// num_ntts: number of NTTs to process
// ntt_size: size of each NTT (must be power of 2)
// stride: distance between start of consecutive NTTs
// omega: primitive root of unity for ntt_size
void cuda_ntt_batched(uint64_t* d_values, uint32_t num_ntts, uint32_t ntt_size, uint32_t stride, uint64_t omega) {
    if (num_ntts == 0) return;

    uint32_t log_n = 0;
    uint32_t temp = ntt_size;
    while (temp > 1) {
        log_n++;
        temp >>= 1;
    }

    uint32_t threads = 256;
    uint32_t total_elements = num_ntts * ntt_size;
    uint32_t total_butterflies = num_ntts * (ntt_size / 2);

    // Bit reversal for all NTTs
    uint32_t blocks = (total_elements + threads - 1) / threads;
    ntt_batched_bit_reverse<<<blocks, threads>>>(d_values, num_ntts, ntt_size, stride, log_n);

    // Butterfly stages for all NTTs
    blocks = (total_butterflies + threads - 1) / threads;
    for (uint32_t stage = 1; stage <= log_n; stage++) {
        ntt_batched_butterfly<<<blocks, threads>>>(d_values, num_ntts, ntt_size, stride, stage, omega);
    }

    cudaDeviceSynchronize();
}

// Batched INTT
void cuda_intt_batched(uint64_t* d_values, uint32_t num_ntts, uint32_t ntt_size, uint32_t stride, uint64_t omega) {
    if (num_ntts == 0) return;

    // Compute omega^-1
    uint64_t inv_omega = bb_pow(omega, ntt_size - 1);

    // NTT with inverse omega
    cuda_ntt_batched(d_values, num_ntts, ntt_size, stride, inv_omega);

    // Scale by 1/n
    uint64_t inv_n = bb_pow(ntt_size, BABYBEAR_PRIME - 2);
    uint32_t threads = 256;
    uint32_t total_elements = num_ntts * ntt_size;
    uint32_t blocks = (total_elements + threads - 1) / threads;
    ntt_batched_scale<<<blocks, threads>>>(d_values, num_ntts, ntt_size, stride, inv_n);

    cudaDeviceSynchronize();
}

// ============================================================================
// FUSED RS ENCODING KERNEL - INTT + zero-pad + NTT in one operation
// ============================================================================

// This kernel takes k-sized data, pads to n-size, ready for NTT
// Input:  d_input[batch_idx * k + i] for i in [0, k)
// Output: d_output[batch_idx * n + i] for i in [0, n), with zeros for i >= k
__global__ void rs_gather_and_pad(
    const uint64_t* d_input,    // Input: num_positions values, each with k elements
    uint64_t* d_output,         // Output: num_positions values, each with n elements
    uint32_t num_positions,
    uint32_t k,
    uint32_t n
) {
    uint32_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total_output = num_positions * n;

    if (global_idx >= total_output) return;

    uint32_t pos_idx = global_idx / n;
    uint32_t elem_idx = global_idx % n;

    if (elem_idx < k) {
        d_output[global_idx] = d_input[pos_idx * k + elem_idx];
    } else {
        d_output[global_idx] = 0;
    }
}

// Scatter encoded values back to shard layout
// Input:  d_encoded[pos_idx * n + shard_idx]
// Output: d_shards[shard_idx * num_positions + pos_idx]
__global__ void rs_scatter_to_shards(
    const uint64_t* d_encoded,
    uint64_t* d_shards,
    uint32_t num_positions,
    uint32_t n
) {
    uint32_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total_elements = num_positions * n;

    if (global_idx >= total_elements) return;

    uint32_t pos_idx = global_idx / n;
    uint32_t shard_idx = global_idx % n;

    d_shards[shard_idx * num_positions + pos_idx] = d_encoded[global_idx];
}

// Full RS encode on GPU:
// 1. Gather k values per position
// 2. INTT (size k)
// 3. Pad to n
// 4. NTT (size n)
// 5. Scatter to n output shards
void cuda_rs_encode(
    const uint64_t* d_input_shards,  // k input shards, each with num_positions elements
    uint64_t* d_output_shards,       // n output shards, each with num_positions elements
    uint64_t* d_work_buffer,         // Workspace: num_positions * max(k_padded, n) elements
    uint32_t num_positions,
    uint32_t k,
    uint32_t n,
    uint64_t omega_k,                // Root of unity for size k (or k_padded)
    uint64_t omega_n                 // Root of unity for size n
) {
    uint32_t threads = 256;

    // Pad k to power of 2
    uint32_t k_padded = 1;
    while (k_padded < k) k_padded <<= 1;

    // Pad n to power of 2 (should already be, but just in case)
    uint32_t n_padded = 1;
    while (n_padded < n) n_padded <<= 1;

    // Step 1: Gather and pad input to k_padded
    // Input layout: d_input_shards[shard_idx * num_positions + pos_idx]
    // Need to transpose to: work[pos_idx * k_padded + shard_idx]
    uint32_t gather_total = num_positions * k_padded;
    uint32_t gather_blocks = (gather_total + threads - 1) / threads;

    // Custom gather kernel for shard layout
    // For now, we'll use a simple approach
    rs_gather_and_pad<<<gather_blocks, threads>>>(d_input_shards, d_work_buffer, num_positions, k, k_padded);

    // Step 2: Batched INTT on all positions (size k_padded each)
    cuda_intt_batched(d_work_buffer, num_positions, k_padded, k_padded, omega_k);

    // Step 3: Pad from k_padded to n_padded and do NTT
    // We need a second buffer or in-place expansion
    // For simplicity, if n_padded > k_padded, we need to re-pad
    if (n_padded > k_padded) {
        // This requires careful buffer management
        // For now, assume k and n are chosen such that we can work in place
        // or use the output buffer
    }

    // Step 4: Batched NTT (size n_padded each)
    cuda_ntt_batched(d_work_buffer, num_positions, n_padded, n_padded, omega_n);

    // Step 5: Scatter to output shards
    uint32_t scatter_total = num_positions * n;
    uint32_t scatter_blocks = (scatter_total + threads - 1) / threads;
    rs_scatter_to_shards<<<scatter_blocks, threads>>>(d_work_buffer, d_output_shards, num_positions, n);

    cudaDeviceSynchronize();
}

} // extern "C"
