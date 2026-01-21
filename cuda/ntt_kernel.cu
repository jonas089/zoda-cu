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
// GPU PADDING KERNEL - Zero-pad from size k to size n efficiently on GPU
// ============================================================================

// Pads batched data from stride_in to stride_out with zeros
// Input:  d_input[batch_idx * stride_in + i] for i in [0, stride_in)
// Output: d_output[batch_idx * stride_out + i] for i in [0, stride_out)
//         where output[i] = input[i] if i < stride_in, else 0
__global__ void gpu_pad_batched(
    const uint64_t* d_input,
    uint64_t* d_output,
    uint32_t num_batches,
    uint32_t stride_in,
    uint32_t stride_out
) {
    uint32_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total_elements = num_batches * stride_out;

    if (global_idx >= total_elements) return;

    uint32_t batch_idx = global_idx / stride_out;
    uint32_t elem_idx = global_idx % stride_out;

    if (elem_idx < stride_in) {
        d_output[global_idx] = d_input[batch_idx * stride_in + elem_idx];
    } else {
        d_output[global_idx] = 0;
    }
}

// Optimized ZODA Reed-Solomon Encoding on GPU
//
// This performs vertical RS extension:
// - Input: num_positions columns, each with k values
// - Output: num_positions columns, each with (k+n) values
//
// Algorithm:
//   For each column position:
//     1. INTT(k) to get polynomial coefficients
//     2. Zero-pad to k+n
//     3. NTT(k+n) to evaluate at k+n points
//
// Memory layout:
//   Input:  [col_0: k values][col_1: k values]...
//   Output: [col_0: k+n values][col_1: k+n values]...
//
// All operations happen on GPU with NO CPU roundtrips for maximum performance
void cuda_rs_encode_vertical(
    const uint64_t* d_input,     // Input: num_positions * ntt_size_k elements
    uint64_t* d_output,          // Output: num_positions * ntt_size_kn elements
    uint64_t* d_intt_work,       // Work buffer: num_positions * ntt_size_k elements
    uint32_t num_positions,      // Number of column positions to process in parallel
    uint32_t ntt_size_k,         // NTT size for k (must be power of 2 >= k)
    uint32_t ntt_size_kn,        // NTT size for k+n (must be power of 2 >= k+n)
    uint64_t omega_k,            // Root of unity for ntt_size_k
    uint64_t omega_kn            // Root of unity for ntt_size_kn
) {
    uint32_t threads = 256;

    // Step 1: Copy input to work buffer (in case d_input and d_intt_work are different)
    cudaMemcpy(d_intt_work, d_input,
               num_positions * ntt_size_k * sizeof(uint64_t),
               cudaMemcpyDeviceToDevice);

    // Step 2: Batched INTT on ALL column positions in parallel
    // This processes num_positions independent NTTs of size ntt_size_k simultaneously
    cuda_intt_batched(d_intt_work, num_positions, ntt_size_k, ntt_size_k, omega_k);

    // Step 3: Zero-pad from ntt_size_k to ntt_size_kn on GPU
    // This avoids a costly GPU->CPU->GPU roundtrip
    uint32_t total_padded = num_positions * ntt_size_kn;
    uint32_t pad_blocks = (total_padded + threads - 1) / threads;
    gpu_pad_batched<<<pad_blocks, threads>>>(
        d_intt_work, d_output,
        num_positions, ntt_size_k, ntt_size_kn
    );

    // Step 4: Batched NTT on ALL column positions in parallel
    // This evaluates the polynomial at k+n points for all columns simultaneously
    cuda_ntt_batched(d_output, num_positions, ntt_size_kn, ntt_size_kn, omega_kn);

    // Single synchronization at the end
    cudaDeviceSynchronize();
}

} // extern "C"
