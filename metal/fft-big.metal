#include <metal_stdlib>
using namespace metal;

#define LIMBS 4   // 256-bit modulus (4 x 64-bit limbs)

// -----------------------------------------------------------
// basic helpers
// -----------------------------------------------------------

// 64-bit wide multiply
inline ulong2 mul_wide(ulong a, ulong b) {
    return ulong2(a * b, mulhi(a, b));
}

inline void clear_n(thread ulong* r, int L) {
    for (int i=0; i<L; ++i) r[i] = 0;
}

inline bool ge_limbs_n(thread const ulong* a,
                       thread const ulong* b,
                       int L) {
    for (int i=L-1; i>=0; --i) {
        if (a[i] > b[i]) return true;
        if (a[i] < b[i]) return false;
    }
    return true;
}

inline void sub_limbs_n(thread ulong* r,
                        thread const ulong* a,
                        thread const ulong* b,
                        int L) {
    ulong borrow = 0;
    for (int i=0; i<L; ++i) {
        ulong sub1 = b[i] + borrow;
        ulong b1   = (sub1 < b[i]);
        ulong tmp  = a[i] - sub1;
        ulong b2   = (a[i] < sub1);
        r[i] = tmp;
        borrow = b1 + b2;
    }
}

inline void add_limbs_n(thread ulong* r,
                        thread const ulong* a,
                        thread const ulong* b,
                        int L) {
    ulong carry = 0;
    for (int i=0; i<L; ++i) {
        ulong s1 = a[i] + b[i];
        ulong c1 = (s1 < a[i]);
        ulong s2 = s1 + carry;
        ulong c2 = (s2 < s1);
        r[i] = s2;
        carry = c1 + c2;
    }
}

inline void add_mod(thread ulong* r,
                    thread const ulong* a,
                    thread const ulong* b,
                    thread const ulong* m) {
    add_limbs_n(r, a, b, LIMBS);
    if (ge_limbs_n(r, m, LIMBS)) {
        sub_limbs_n(r, r, m, LIMBS);
    }
}

inline void sub_mod(thread ulong* r,
                    thread const ulong* a,
                    thread const ulong* b,
                    thread const ulong* m) {
    ulong borrow = 0;
    for (int i=0; i<LIMBS; ++i) {
        ulong sub1 = b[i] + borrow;
        ulong b1   = (sub1 < b[i]);
        ulong tmp  = a[i] - sub1;
        ulong b2   = (a[i] < sub1);
        r[i] = tmp;
        borrow = b1 + b2;
    }
    if (borrow) {
        ulong carry = 0;
        for (int i=0; i<LIMBS; ++i) {
            ulong s1 = r[i] + m[i];
            ulong c1 = (s1 < r[i]);
            ulong s2 = s1 + carry;
            ulong c2 = (s2 < s1);
            r[i] = s2;
            carry = c1 + c2;
        }
    }
    if (ge_limbs_n(r, m, LIMBS)) {
        sub_limbs_n(r, r, m, LIMBS);
    }
}

// -----------------------------------------------------------
// Montgomery multiplication: r = a*b*R^-1 mod m
// -----------------------------------------------------------
inline void mont_mul(thread ulong*       r,
                     thread const ulong* a,
                     thread const ulong* b,
                     thread const ulong* m,
                     ulong nprime) {
    const int K = LIMBS;
    thread ulong t[2*K];
    clear_n(t, 2*K);

    // multiply
    for (int i=0; i<K; i++) {
        ulong carry = 0;
        for (int j=0; j<K; j++) {
            ulong2 prod = mul_wide(a[i], b[j]);
            ulong sum = t[i+j] + prod.x;
            ulong c1 = (sum < t[i+j]);
            sum += carry;
            ulong c2 = (sum < carry);
            t[i+j] = sum;
            carry = prod.y + c1 + c2;
        }
        ulong sum = t[i+K] + carry;
        ulong c1 = (sum < t[i+K]);
        t[i+K] = sum;
        if (c1 && i+K+1 < 2*K) {
            t[i+K+1] += 1;
        }
    }

    // Montgomery reduction
    for (int i=0; i<K; i++) {
        ulong u = t[i] * nprime;
        ulong carry = 0;
        for (int j=0; j<K; j++) {
            ulong2 prod = mul_wide(u, m[j]);
            ulong sum = t[i+j] + prod.x;
            ulong c1 = (sum < t[i+j]);
            sum += carry;
            ulong c2 = (sum < carry);
            t[i+j] = sum;
            carry = prod.y + c1 + c2;
        }
        int idx = i+K;
        while (carry) {
            ulong sum = t[idx] + carry;
            carry = (sum < t[idx]);
            t[idx] = sum;
            idx++;
        }
    }

    for (int i=0; i<K; i++) {
        r[i] = t[i+K];
    }

    if (ge_limbs_n(r, m, K)) {
        sub_limbs_n(r, r, m, K);
    }
}

inline void mul_mod(thread ulong*       r,
                    thread const ulong* a,
                    thread const ulong* b,
                    thread const ulong* m,
                    ulong nprime) {
    mont_mul(r, a, b, m, nprime);
}

// -----------------------------------------------------------
// FFT stage (DIT): expects bit-reversed input; outputs natural order
// -----------------------------------------------------------
kernel void fft_stage(
    device ulong*       data,
    device const ulong* twiddles,
    constant uint&      n,
    constant uint&      len,
    constant ulong*     modulus,
    constant ulong&     nprime,
    uint tid [[thread_position_in_grid]]
) {
    uint half_len = len >> 1;
    uint group    = tid / half_len;
    uint pos      = tid % half_len;
    uint i        = group * len + pos;
    if (i + half_len >= n) return;

    uint step = n / len;

    thread ulong m[LIMBS];
    for (int k=0; k<LIMBS; ++k) m[k] = modulus[k];

    thread ulong w[LIMBS];
    uint tw_idx = pos * step;
    for (int k=0; k<LIMBS; ++k) {
        w[k] = twiddles[tw_idx * LIMBS + k];
    }

    thread ulong u[LIMBS], v[LIMBS];
    for (int k=0; k<LIMBS; ++k) {
        u[k] = data[i*LIMBS + k];
        v[k] = data[(i+half_len)*LIMBS + k];
    }

    thread ulong vtmp[LIMBS];
    mul_mod(vtmp, v, w, m, nprime);

    thread ulong a[LIMBS], b[LIMBS];
    add_mod(a, u, vtmp, m);
    sub_mod(b, u, vtmp, m);

    for (int k=0; k<LIMBS; ++k) {
        data[i*LIMBS + k]         = a[k];
        data[(i+half_len)*LIMBS + k] = b[k];
    }
}

// -----------------------------------------------------------
// Bit-reversal permutation kernel
// -----------------------------------------------------------
kernel void bitrev_permute(
    device ulong* data,
    constant uint& n,
    constant uint& logn,
    uint tid [[thread_position_in_grid]]
) {
    uint i = tid;
    if (i >= n) return;

    uint j = 0;
    uint t = i;
    for (uint b = 0; b < logn; ++b) {
        j = (j << 1) | (t & 1u);
        t >>= 1;
    }
    if (i >= j) return;

    for (int k = 0; k < LIMBS; ++k) {
        ulong tmp                 = data[i*LIMBS + k];
        data[i*LIMBS + k]         = data[j*LIMBS + k];
        data[j*LIMBS + k]         = tmp;
    }
}
