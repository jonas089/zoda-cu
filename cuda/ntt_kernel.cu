#include <cuda_runtime.h>
#include <stdint.h>

#define P 2013265921u  // BabyBear prime: 2^31 - 2^27 + 1
#define THREADS 256      // columns per block
#define CHUNK_COLS 1024  // columns per chunk; device memory per stream is n * CHUNK_COLS * 4 bytes
#define NUM_STREAMS 3    // upload of chunk i+1, compute of chunk i, download of chunk i-1

// ---------------------------------------------------------------------------
// Skeleton. Plumbing is done; the kernel bodies are yours.
//
// Layout: row-major values[n rows][cols]. Every column is an independent
// n-point transform. One thread owns one column, so neighbouring threads touch
// neighbouring addresses and every twiddle load is the same across a block.
// No shared memory anywhere: no thread ever needs another thread's data.
//
// Orchestration: columns are processed in chunks of CHUNK_COLS. Each chunk is
// uploaded once, transformed by one kernel launch per pass, and downloaded
// once, all queued on one stream. Chunks rotate over NUM_STREAMS streams so
// the copies of one chunk overlap the kernels of another. Inside a pass the L
// stages run on 2^L values held in registers; the kernel boundary is the
// barrier between passes. The only state between launches is s0.
// ---------------------------------------------------------------------------

// 1. Field arithmetic in Montgomery form, branch-free (add, sub, mont_mul).
//    TODO

// 2. Elementwise conversion kernels: to Montgomery form on the way in, back
//    out (optionally times a scale, for n^-1) on the way out.
//    TODO

// 3. The pass kernel. Stages s0 .. s0+L-1 of an n-point NTT on every column
//    of the chunk. Stage s pairs rows differing in bit s-1. This pass touches
//    row bits s0-1 .. s0+L-2, so a thread's group is the 2^L rows agreeing on
//    all other bits: row = base + t * 2^(s0-1), t = 0 .. 2^L-1. blockIdx.y
//    picks the group, threads pick columns.
template <int L>
__global__ void ntt_pass(uint32_t* buf, uint32_t cols, uint32_t log_n, uint32_t s0,
                         const uint32_t* roots) {
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    uint32_t g = blockIdx.y;

    // TODO: base, step and j0 from g and s0.

    uint32_t x[1 << L];
    // TODO: load x[t] = buf[(base + t*step) * cols + col]

    // TODO: L stages. Stage k pairs x[t] with x[t | (1 << k)] for t with bit k
    //       clear. Twiddle index j = j0 + (t & ((1 << k) - 1)) * step, and
    //       w = roots[j << (log_n - s0 - k)]. Fully unroll so x stays in registers.

    // TODO: store x back.
}

// 4. Stage grouping: log_n stages into passes of at most MAX_L stages.
#define MAX_L 6
struct PassPlan { int count; int L[8]; };
static PassPlan plan_passes(uint32_t log_n) {
    PassPlan p;
    p.count = (log_n + MAX_L - 1) / MAX_L;
    int base = log_n / p.count, rem = log_n % p.count;
    for (int i = 0; i < p.count; i++) p.L[i] = base + (i < rem ? 1 : 0);
    return p;
}

static cudaError_t launch_pass(int L, uint32_t* buf, uint32_t cols, uint32_t log_n,
                               uint32_t s0, const uint32_t* roots, cudaStream_t stream) {
    dim3 grid((cols + THREADS - 1) / THREADS, (1u << log_n) >> L), block(THREADS);
    switch (L) {
#define C(l) case l: ntt_pass<l><<<grid, block, 0, stream>>>(buf, cols, log_n, s0, roots); break;
        C(1) C(2) C(3) C(4) C(5) C(6)
#undef C
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

// 5. Host entry. values: [n rows][cols] row-major, transformed in place.
//    roots: n forward roots omega^i.
extern "C" cudaError_t cuda_ntt(uint32_t* values, const uint32_t* roots, uint32_t n, uint32_t cols) {
    uint32_t log_n = 0;
    while ((1u << log_n) < n) log_n++;
    size_t host_pitch = (size_t)cols * sizeof(uint32_t);

    // Pin the caller's buffer for the duration of the call. Async copies from
    // pageable memory silently become synchronous, which would kill the overlap.
    cudaHostRegister(values, (size_t)n * cols * sizeof(uint32_t), cudaHostRegisterDefault);

    uint32_t* d_roots;
    cudaMalloc((void**)&d_roots, n * sizeof(uint32_t));
    cudaMemcpy(d_roots, roots, n * sizeof(uint32_t), cudaMemcpyHostToDevice);
    // TODO: to-Montgomery kernel on d_roots.

    cudaStream_t streams[NUM_STREAMS];
    uint32_t* d_buf[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; s++) {
        cudaStreamCreate(&streams[s]);
        cudaMalloc((void**)&d_buf[s], (size_t)n * CHUNK_COLS * sizeof(uint32_t));
    }

    cudaError_t err = cudaSuccess;
    int i = 0;
    for (uint32_t c0 = 0; c0 < cols && err == cudaSuccess; c0 += CHUNK_COLS, i++) {
        uint32_t w = cols - c0 < CHUNK_COLS ? cols - c0 : CHUNK_COLS;
        size_t dev_pitch = (size_t)w * sizeof(uint32_t);
        cudaStream_t st = streams[i % NUM_STREAMS];
        uint32_t* buf = d_buf[i % NUM_STREAMS];  // reuse is safe: same-stream order puts its previous download first

        cudaMemcpy2DAsync(buf, dev_pitch, values + c0, host_pitch, dev_pitch, n, cudaMemcpyHostToDevice, st);

        // TODO: to-Montgomery kernel on buf.
        // TODO: bit-reverse the rows (DIT wants bit-reversed input), or run the
        //       inverse as DIF and the forward as DIT so no permutation is needed.

        PassPlan pp = plan_passes(log_n);
        uint32_t s0 = 1;
        for (int k = 0; k < pp.count && err == cudaSuccess; k++) {
            err = launch_pass(pp.L[k], buf, w, log_n, s0, d_roots, st);
            s0 += pp.L[k];
        }

        // TODO: from-Montgomery kernel on buf.

        cudaMemcpy2DAsync(values + c0, host_pitch, buf, dev_pitch, dev_pitch, n, cudaMemcpyDeviceToHost, st);
    }

    cudaDeviceSynchronize();
    for (int s = 0; s < NUM_STREAMS; s++) {
        cudaStreamDestroy(streams[s]);
        cudaFree(d_buf[s]);
    }
    cudaFree(d_roots);
    cudaHostUnregister(values);
    return err != cudaSuccess ? err : cudaGetLastError();
}

extern "C" cudaError_t cuda_intt(uint32_t* values, const uint32_t* roots, uint32_t n, uint32_t cols) {
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

    cudaError_t err = cuda_ntt(values, inv_roots, n, cols);

    // TODO: fold this scale into the from-Montgomery kernel instead of a host pass.
    for (size_t i = 0; i < (size_t)n * cols; i++) {
        values[i] = values[i] * inv_n % P;
    }

    delete[] inv_roots;
    return err;
}
