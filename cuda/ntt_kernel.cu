#include <cuda_runtime.h>
#include <stdint.h>

#define P 2013265921u  // BabyBear prime: 2^31 - 2^27 + 1


extern "C" cudaError_t cuda_ntt(uint32_t* values, const uint32_t* roots, uint32_t n) {
    size_t bytes = n * sizeof(uint32_t);
    uint32_t log_n = 0;
    while ((1u << log_n) < n) log_n++;

    uint32_t *d_values, *d_roots;
    cudaMalloc((void**)&d_values, bytes);
    cudaMalloc((void**)&d_roots, bytes);
    cudaMemcpy(d_values, values, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_roots, roots, bytes, cudaMemcpyHostToDevice);

    // 1. Run the NTT butterfly
    // 2. Montgomery reduction and arithmetic
    // 3. Optimize for batching
    // 4. Optimize for shared memory
    
    cudaDeviceSynchronize();
    cudaMemcpy(values, d_values, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_values);
    cudaFree(d_roots);
    return cudaGetLastError();
}

extern "C" cudaError_t cuda_intt(uint32_t* values, const uint32_t* roots, uint32_t n) {
    // Backwards circle: omega^-i is the same point as omega^(n-i).
    uint32_t* inv_roots = new uint32_t[n];
    inv_roots[0] = roots[0];
    for (uint32_t i = 1; i < n; i++) {
        inv_roots[i] = roots[n - i];
    }

    // n^-1 = (2^-1)^log2(n), and 2^-1 mod P is (P+1)/2
    uint64_t inv_n = 1;
    for (uint32_t m = n; m > 1; m /= 2) {
        inv_n = inv_n * ((P + 1) / 2) % P;
    }

    cudaError_t err = cuda_ntt(values, inv_roots, n);

    for (uint32_t i = 0; i < n; i++) {
        values[i] = values[i] * inv_n % P;
    }

    delete[] inv_roots;
    return err;
}
