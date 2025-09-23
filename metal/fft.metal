#include <metal_stdlib>
using namespace metal;

inline uint32_t mod_add(uint32_t a, uint32_t b, uint32_t p) {
    uint32_t s = a + b;
    return (s >= p) ? s - p : s;
}

inline uint32_t mod_sub(uint32_t a, uint32_t b, uint32_t p) {
    return (a >= b) ? a - b : a + p - b;
}

inline uint32_t mod_mul(uint32_t a, uint32_t b, uint32_t p) {
    ulong t = (ulong)a * (ulong)b;
    return (uint32_t)(t % p);
}

kernel void fft_stage(
    device uint32_t*       data      [[buffer(0)]],
    device const uint32_t* twiddles  [[buffer(1)]],
    constant uint32_t&     n         [[buffer(2)]],
    constant uint32_t&     len       [[buffer(3)]],
    constant uint32_t&     modulus   [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint half_len = len >> 1;
    uint group    = tid / half_len;
    uint pos      = tid % half_len;
    uint i        = group * len + pos;
    if (i + half_len >= n) return;

    uint step = n / len;
    uint32_t w = twiddles[(pos * step) % n];

    uint32_t u = data[i];
    uint32_t v = mod_mul(data[i + half_len], w, modulus);

    data[i]          = mod_add(u, v, modulus);
    data[i + half_len] = mod_sub(u, v, modulus);
}
