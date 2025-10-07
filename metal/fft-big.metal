#include <metal_stdlib>
using namespace metal;

#define LIMBS 4               // 256-bit modulus (4 x 64-bit limbs)
#define MAX_SHARED_SIZE 256   // Optimized for M1/M2/M3 - doubles/triples active threadgroups per SM

// Vectorized 256-bit field element (4x64-bit limbs)
using fe_t = ulong4;

// -----------------------------------------------------------
// basic helpers (unchanged numerically)
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
/* Montgomery multiplication: r = a*b*R^-1 mod m (unchanged math) */
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

// -----------------------------------------------------------
// Shared Memory FFT Kernel - coalesced via ulong4 with batching support
// -----------------------------------------------------------
kernel void fft_shared_memory(
    device fe_t*        data,           // Input/output data array (batch_size * N elements)
    constant uint&      n,              // Size of each FFT (elements)
    constant uint&      batch_size,     // Number of FFTs in batch
    device const fe_t*  twiddles,       // Twiddle factors as fe_t
    constant ulong*     modulus,        // Modulus (4 limbs)
    constant ulong&     nprime,         // Montgomery parameter
    uint tid [[thread_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]
) {
    // Use threadgroup memory: one fe_t per element
    threadgroup fe_t shared_data[MAX_SHARED_SIZE];

    uint block_size = min(n, uint(MAX_SHARED_SIZE));
    uint num_blocks_per_fft = (n + block_size - 1) / block_size;
    uint total_blocks = num_blocks_per_fft * batch_size;
    
    if (group_id >= total_blocks) return;
    
    // Determine which FFT and which block within that FFT
    uint fft_id = group_id / num_blocks_per_fft;
    uint block_id_in_fft = group_id % num_blocks_per_fft;
    
    // Calculate offset for this FFT in the data array
    uint fft_offset = fft_id * n;
    uint block_start = fft_offset + block_id_in_fft * block_size;
    uint block_end = min(block_start + block_size, fft_offset + n);
    uint actual_block_size = block_end - block_start;

    // Load modulus into registers
    thread ulong m[LIMBS];
    for (int k = 0; k < LIMBS; ++k) m[k] = modulus[k];

    // Coalesced load: one 32B read per element
    if (local_id < actual_block_size) {
        uint g = block_start + local_id;
        shared_data[local_id] = data[g];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // log2(actual_block_size)
    uint log_block_size = 0;
    for (uint t = actual_block_size; t > 1; t >>= 1) ++log_block_size;

    uint twiddle_offset = 0;

    for (uint stage = 0; stage < log_block_size; ++stage) {
        uint step = 1u << stage;
        uint groups_in_block = actual_block_size >> (stage + 1);

        if (local_id < groups_in_block * step) {
            uint group = local_id / step;
            uint pos   = local_id % step;
            uint i = group * (step << 1) + pos;
            uint j = i + step;

            if (j < actual_block_size) {
                // Load twiddle with correct stage indexing
                fe_t wv = twiddles[(1u << stage) - 1u + pos];
                thread ulong w[LIMBS];
                w[0] = wv.x; w[1] = wv.y; w[2] = wv.z; w[3] = wv.w;

                // Load data (vectors) and unpack to limbs
                fe_t ui = shared_data[i];
                fe_t vj = shared_data[j];
                thread ulong u[LIMBS], v[LIMBS];
                u[0] = ui.x; u[1] = ui.y; u[2] = ui.z; u[3] = ui.w;
                v[0] = vj.x; v[1] = vj.y; v[2] = vj.z; v[3] = vj.w;

                // Butterfly
                thread ulong temp[LIMBS], un[LIMBS], vn[LIMBS];
                mont_mul(temp, v, w, m, nprime);  // temp = v * w (Montgomery)
                add_mod(un, u, temp, m);          // u' = u + temp
                sub_mod(vn, u, temp, m);          // v' = u - temp

                // Pack back and store
                shared_data[i] = fe_t{un[0], un[1], un[2], un[3]};
                shared_data[j] = fe_t{vn[0], vn[1], vn[2], vn[3]};
            }
        }

        twiddle_offset += step;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Coalesced store: one 32B write per element
    if (local_id < actual_block_size) {
        uint g = block_start + local_id;
        data[g] = shared_data[local_id];
    }
}

// -----------------------------------------------------------
// Simple Butterfly Kernel - coalesced via ulong4 with batching support
// -----------------------------------------------------------
kernel void butterfly_fft(
    device fe_t*        data,           // Input/output data array (batch_size * N elements)
    constant uint&      n,              // Size of each FFT
    constant uint&      batch_size,     // Number of FFTs in batch
    constant uint&      stage,          // Current stage (0, 1, 2, ...)
    device const fe_t*  twiddles,       // Twiddle factors for this stage
    constant ulong*     modulus,        // Modulus
    constant ulong&     nprime,         // Montgomery parameter
    uint tid [[thread_position_in_grid]]
) {
    uint step       = 1u << stage;            // Distance between butterfly elements
    uint num_groups = n >> (stage + 1);       // Number of groups in this stage
    uint threads_per_fft = num_groups * step;
    uint total_threads = threads_per_fft * batch_size;

    if (tid >= total_threads) return;

    // Determine which FFT this thread belongs to
    uint fft_id = tid / threads_per_fft;
    uint tid_in_fft = tid % threads_per_fft;
    
    // Calculate offset for this FFT in the data array
    uint fft_offset = fft_id * n;

    // Butterfly indices within this FFT
    uint group = tid_in_fft / step;
    uint pos   = tid_in_fft % step;
    uint i     = fft_offset + group * (step << 1) + pos;
    uint j     = i + step;
    if (j >= fft_offset + n) return;

    // Load modulus
    thread ulong m[LIMBS];
    for (int k = 0; k < LIMBS; ++k) m[k] = modulus[k];

    // Twiddle with correct stage indexing
    fe_t wv = twiddles[(1u << stage) - 1u + pos];
    thread ulong w[LIMBS];
    w[0] = wv.x; w[1] = wv.y; w[2] = wv.z; w[3] = wv.w;

    // Data (vectors) -> limbs
    fe_t ui = data[i];
    fe_t vj = data[j];
    thread ulong u[LIMBS], v[LIMBS];
    u[0] = ui.x; u[1] = ui.y; u[2] = ui.z; u[3] = ui.w;
    v[0] = vj.x; v[1] = vj.y; v[2] = vj.z; v[3] = vj.w;

    // Butterfly math
    thread ulong temp[LIMBS], un[LIMBS], vn[LIMBS];
    mont_mul(temp, v, w, m, nprime);
    add_mod(un, u, temp, m);
    sub_mod(vn, u, temp, m);

    // Store back (packed)
    data[i] = fe_t{un[0], un[1], un[2], un[3]};
    data[j] = fe_t{vn[0], vn[1], vn[2], vn[3]};
}

// -----------------------------------------------------------
// Bit-reversal permutation kernel (swap full elements as fe_t) with batching support
// -----------------------------------------------------------
kernel void bitrev_permute(
    device fe_t*  data,
    constant uint& n,
    constant uint& batch_size,
    constant uint& logn,
    uint tid [[thread_position_in_grid]]
) {
    uint total_elements = n * batch_size;
    if (tid >= total_elements) return;
    
    // Determine which FFT this thread belongs to
    uint fft_id = tid / n;
    uint i_in_fft = tid % n;
    uint fft_offset = fft_id * n;
    
    uint i = fft_offset + i_in_fft;

    uint j_in_fft = 0;
    uint t = i_in_fft;
    for (uint b = 0; b < logn; ++b) {
        j_in_fft = (j_in_fft << 1) | (t & 1u);
        t >>= 1;
    }
    
    uint j = fft_offset + j_in_fft;
    
    if (i_in_fft >= j_in_fft) return;

    // Swap entire 256-bit elements in one go
    fe_t tmp = data[i];
    data[i]  = data[j];
    data[j]  = tmp;
}
