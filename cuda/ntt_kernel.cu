#include <cuda_runtime.h>
#include <stdint.h>

// BabyBear prime: 2^31 - 2^27 + 1
#define BABYBEAR_PRIME 2013265921ULL

// BabyBear field element
typedef uint64_t BabyBearElem;

// Device functions for BabyBear field arithmetic
__device__ __forceinline__ uint64_t bb_reduce(uint64_t val) {
    if (val >= BABYBEAR_PRIME) {
        val -= BABYBEAR_PRIME;
    }
    if (val >= BABYBEAR_PRIME) {
        val -= BABYBEAR_PRIME;
    }
    return val;
}

__device__ __forceinline__ uint64_t bb_add(uint64_t a, uint64_t b) {
    uint64_t sum = a + b;
    return bb_reduce(sum);
}

__device__ __forceinline__ uint64_t bb_sub(uint64_t a, uint64_t b) {
    if (a >= b) {
        return a - b;
    } else {
        return a + BABYBEAR_PRIME - b;
    }
}

__device__ __forceinline__ uint64_t bb_mul(uint64_t a, uint64_t b) {
    // Use 128-bit intermediate to avoid overflow
    unsigned long long product_hi, product_lo;
    product_lo = a * b;
    product_hi = __umul64hi(a, b);

    // Fast reduction for BabyBear prime
    // Since BABYBEAR_PRIME = 2^31 - 2^27 + 1
    // We can use Barrett reduction or direct modulo
    // For now, use direct modulo (compiler will optimize)
    uint64_t result = product_lo % BABYBEAR_PRIME;
    return result;
}

// Modular exponentiation on device
__device__ uint64_t bb_pow(uint64_t base, uint64_t exp) {
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

} // extern "C"
