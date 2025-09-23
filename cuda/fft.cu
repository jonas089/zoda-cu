extern "C" {

    #include <stdint.h>
    
    __device__ __forceinline__ uint32_t mod_add(uint32_t a, uint32_t b, uint32_t p) {
        uint32_t s = a + b;
        return (s >= p) ? s - p : s;
    }
    
    __device__ __forceinline__ uint32_t mod_sub(uint32_t a, uint32_t b, uint32_t p) {
        return (a >= b) ? a - b : a + p - b;
    }
    
    __device__ __forceinline__ uint32_t mod_mul(uint32_t a, uint32_t b, uint32_t p) {
        unsigned long long t = (unsigned long long)a * b;
        return (uint32_t)(t % p);
    }
    
    /// One FFT stage kernel
    __global__ void fft_stage(uint32_t *data,
                              const uint32_t *twiddles,
                              int n,
                              int len,
                              uint32_t p)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int half = len >> 1;
        int group = tid / half;
        int pos   = tid % half;
    
        int i = group * len + pos;
        if (i + half >= n) return;
    
        int step = n / len;
        uint32_t w = twiddles[(pos * step) % n];
    
        uint32_t u = data[i];
        uint32_t v = mod_mul(data[i + half], w, p);
    
        data[i]        = mod_add(u, v, p);
        data[i + half] = mod_sub(u, v, p);
    }
    
    } // extern "C"
    