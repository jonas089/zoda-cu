#include <cuda_runtime.h>
#include <stdint.h>

// GPU version of src/ntt/cpu.rs::ntt for many columns at once.
//
// One flat array holds every column, row-major: row r of column c is at
// values[r * cols + c]. Column c is one CPU `coeffs` vector with stride cols.
//
//        col 0  col 1  col 2
// row 0    [0]    [1]    [2]
// row 1    [3]    [4]    [5]
// row 2    [6]    [7]    [8]
// row 3    [9]   [10]   [11]

#define P          2013265921u   // the CPU's `modulus`
#define P_NEG_INV  0x77FFFFFFu   // -P^-1 mod 2^32
#define R2_MOD_P   0x45DDDDE3u   // 2^64 mod P
#define THREADS    256           // columns per thread block
#define CHUNK_COLS 1024          // columns uploaded per chunk
#define NUM_STREAMS 3            // chunks in flight at once

__device__ inline size_t idx(uint32_t row, uint32_t col, uint32_t cols) {
    return (size_t)row * cols + col;
}

// ---- field arithmetic ------------------------------------------------------
// Values live in Montgomery form (x * 2^32 mod P) so that mul needs no division.
// add/sub are the CPU's, with min() replacing the if.

__device__ inline uint32_t bb_add(uint32_t a, uint32_t b) {   // (a + b) % P
    uint32_t s = a + b;
    return min(s, s - P);
}
__device__ inline uint32_t bb_sub(uint32_t a, uint32_t b) {   // (a + P - b) % P
    uint32_t d = a - b;
    return min(d, d + P);
}
__device__ inline uint32_t mont_redc(uint32_t lo, uint32_t hi) {   // (hi:lo) / 2^32 mod P
    uint32_t m = lo * P_NEG_INV;
    uint32_t t = hi + __umulhi(m, P) + (lo != 0);
    return min(t, t - P);
}
__device__ inline uint32_t mont_mul(uint32_t a, uint32_t b) {   // (a * b) % P, Montgomery form
    return mont_redc(a * b, __umulhi(a, b));
}

__global__ void to_mont_kernel(uint32_t* v, uint32_t total) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) v[i] = mont_mul(v[i], R2_MOD_P);
}
// Leaves Montgomery form and multiplies by a plain scale (1, or n^-1 for the inverse).
__global__ void from_mont_kernel(uint32_t* v, uint32_t total, uint32_t scale) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) v[i] = mont_redc(v[i] * scale, __umulhi(v[i], scale));
}

// ---- bit reversal. CPU: `let mut coeffs = reverse(values);` ----------------
__global__ void bit_reverse_kernel(uint32_t* coeffs, uint32_t cols, uint32_t total, uint32_t log_n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;   // one thread per element
    if (i >= total) return;
    uint32_t row = i / cols, col = i % cols;
    uint32_t rev = __brev(row) >> (32 - log_n);
    if (row < rev) {
        uint32_t* a = &coeffs[idx(row, col, cols)];
        uint32_t* b = &coeffs[idx(rev, col, cols)];
        uint32_t t = *a; *a = *b; *b = t;
    }
}

// ---- one stage. CPU: the body of `while len <= n` ---------------------------
// The CPU visits the n/2 butterflies of a stage as (block, j) pairs:
//     for block in coeffs.chunks_mut(len), for j in 0..half
//     lo = block * len + j,  hi = lo + half
// blockIdx.y counts those pairs; the thread index picks the column.

struct Butterfly { uint32_t lo, hi, j; };

__device__ inline Butterfly butterfly_rows(uint32_t b, uint32_t len) {
    uint32_t half  = len / 2;
    uint32_t block = b / half;
    uint32_t j     = b % half;
    Butterfly bf;
    bf.lo = block * len + j;
    bf.hi = bf.lo + half;
    bf.j  = j;
    return bf;
}

// One thread = one butterfly of one column.
__global__ void ntt_stage(uint32_t* coeffs, uint32_t cols, uint32_t n, uint32_t len,
                          const uint32_t* roots) {
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    Butterfly bf = butterfly_rows(blockIdx.y, len);
    uint32_t stride = n / len;

    uint32_t* lo = &coeffs[idx(bf.lo, col, cols)];   // CPU: lo[j]
    uint32_t* hi = &coeffs[idx(bf.hi, col, cols)];   // CPU: hi[j]

    uint32_t w = roots[bf.j * stride];               // CPU: let w = roots[j*stride];
    uint32_t a = *lo;                                // CPU: let a = lo[j];
    uint32_t b = mont_mul(*hi, w);                   // CPU: let b = hi[j] * w % modulus;
    *lo = bb_add(a, b);                              // CPU: lo[j] = (a + b) % modulus;
    *hi = bb_sub(a, b);                              // CPU: hi[j] = (a + modulus - b) % modulus;
}

// ---- entry points -------------------------------------------------------------
// values: [n rows][cols], every column transformed in place. Columns go through
// in chunks; each chunk is uploaded once, gets one launch per stage, and is
// downloaded once, on its own stream. scale is applied on the way out.

static cudaError_t ntt_all_columns(uint32_t* values, const uint32_t* roots, uint32_t n,
                                   uint32_t cols, uint32_t scale) {
    uint32_t log_n = 0;
    while ((1u << log_n) < n) log_n++;
    size_t row_bytes = (size_t)cols * sizeof(uint32_t);
    uint32_t col_blocks = (CHUNK_COLS + THREADS - 1) / THREADS;

    // Pinned host memory is required for the async copies to overlap.
    cudaHostRegister(values, (size_t)n * row_bytes, cudaHostRegisterDefault);

    uint32_t* d_roots;
    cudaMalloc((void**)&d_roots, n * sizeof(uint32_t));
    cudaMemcpy(d_roots, roots, n * sizeof(uint32_t), cudaMemcpyHostToDevice);
    to_mont_kernel<<<(n + THREADS - 1) / THREADS, THREADS>>>(d_roots, n);

    cudaStream_t streams[NUM_STREAMS];
    uint32_t* d_coeffs[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; s++) {
        cudaStreamCreate(&streams[s]);
        cudaMalloc((void**)&d_coeffs[s], (size_t)n * CHUNK_COLS * sizeof(uint32_t));
    }

    int chunk = 0;
    for (uint32_t c0 = 0; c0 < cols; c0 += CHUNK_COLS, chunk++) {
        uint32_t w = cols - c0 < CHUNK_COLS ? cols - c0 : CHUNK_COLS;
        size_t chunk_row_bytes = (size_t)w * sizeof(uint32_t);
        uint32_t total = n * w;
        cudaStream_t st = streams[chunk % NUM_STREAMS];
        uint32_t* coeffs = d_coeffs[chunk % NUM_STREAMS];

        cudaMemcpy2DAsync(coeffs, chunk_row_bytes, values + c0, row_bytes, chunk_row_bytes, n,
                          cudaMemcpyHostToDevice, st);

        to_mont_kernel<<<(total + THREADS - 1) / THREADS, THREADS, 0, st>>>(coeffs, total);
        bit_reverse_kernel<<<(total + THREADS - 1) / THREADS, THREADS, 0, st>>>(coeffs, w, total, log_n);
        for (uint32_t len = 2; len <= n; len *= 2) {
            ntt_stage<<<dim3(col_blocks, n / 2), THREADS, 0, st>>>(coeffs, w, n, len, d_roots);
        }
        from_mont_kernel<<<(total + THREADS - 1) / THREADS, THREADS, 0, st>>>(coeffs, total, scale);

        cudaMemcpy2DAsync(values + c0, row_bytes, coeffs, chunk_row_bytes, chunk_row_bytes, n,
                          cudaMemcpyDeviceToHost, st);
    }

    cudaDeviceSynchronize();
    for (int s = 0; s < NUM_STREAMS; s++) {
        cudaStreamDestroy(streams[s]);
        cudaFree(d_coeffs[s]);
    }
    cudaFree(d_roots);
    cudaHostUnregister(values);
    return cudaGetLastError();
}

extern "C" cudaError_t cuda_ntt(uint32_t* values, const uint32_t* roots, uint32_t n, uint32_t cols) {
    return ntt_all_columns(values, roots, n, cols, 1);
}

// CPU: intt = ntt with inv_roots, then multiply by n^-1.
extern "C" cudaError_t cuda_intt(uint32_t* values, const uint32_t* roots, uint32_t n, uint32_t cols) {
    uint32_t* inv_roots = new uint32_t[n];
    inv_roots[0] = roots[0];
    for (uint32_t i = 1; i < n; i++) inv_roots[i] = roots[n - i];

    uint64_t inv_n = 1;                       // (2^-1)^log2(n), with 2^-1 = (P+1)/2
    for (uint32_t m = n; m > 1; m /= 2) inv_n = inv_n * ((P + 1) / 2) % P;

    cudaError_t err = ntt_all_columns(values, inv_roots, n, cols, (uint32_t)inv_n);
    delete[] inv_roots;
    return err;
}
