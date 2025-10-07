#include <metal_stdlib>
using namespace metal;

#define LIMBS 4   // 256-bit modulus (4 x 64-bit limbs)
#define MAX_SHARED_SIZE 1024  // Maximum elements in shared memory

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

// -----------------------------------------------------------
// Shared Memory FFT Kernel - cuFFT style
// -----------------------------------------------------------
kernel void fft_shared_memory(
    device ulong*       data,           // Input/output data array
    constant uint&      n,              // Total size of FFT
    device const ulong* twiddles,       // All twiddle factors (organized by stage)
    constant ulong*     modulus,        // Modulus
    constant ulong&     nprime,         // Montgomery parameter
    uint tid [[thread_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]
) {
    // Use threadgroup memory (shared memory equivalent)
    threadgroup ulong shared_data[MAX_SHARED_SIZE * LIMBS];
    
    uint block_size = min(n, uint(MAX_SHARED_SIZE));
    uint num_blocks = (n + block_size - 1) / block_size;
    
    if (group_id >= num_blocks) return;
    
    uint block_start = group_id * block_size;
    uint block_end = min(block_start + block_size, n);
    uint actual_block_size = block_end - block_start;
    
    // Load modulus into thread memory
    thread ulong m[LIMBS];
    for (int k = 0; k < LIMBS; ++k) {
        m[k] = modulus[k];
    }
    
    // Load data into shared memory
    if (local_id < actual_block_size) {
        uint global_idx = block_start + local_id;
        for (int k = 0; k < LIMBS; ++k) {
            shared_data[local_id * LIMBS + k] = data[global_idx * LIMBS + k];
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Perform FFT stages in shared memory
    uint log_block_size = 0;
    uint temp_size = actual_block_size;
    while (temp_size > 1) {
        log_block_size++;
        temp_size >>= 1;
    }
    
    uint twiddle_offset = 0;
    
    for (uint stage = 0; stage < log_block_size; ++stage) {
        uint step = 1u << stage;
        uint num_groups_in_block = actual_block_size >> (stage + 1);
        
        if (local_id < num_groups_in_block * step) {
            uint group = local_id / step;
            uint pos = local_id % step;
            uint i = group * (step << 1) + pos;
            uint j = i + step;
            
            if (j < actual_block_size) {
                // Load twiddle factor
                thread ulong w[LIMBS];
                for (int k = 0; k < LIMBS; ++k) {
                    w[k] = twiddles[(twiddle_offset + pos) * LIMBS + k];
                }
                
                // Load data from shared memory
                thread ulong u[LIMBS], v[LIMBS];
                for (int k = 0; k < LIMBS; ++k) {
                    u[k] = shared_data[i * LIMBS + k];
                    v[k] = shared_data[j * LIMBS + k];
                }
                
                // Butterfly operation
                thread ulong temp[LIMBS];
                mont_mul(temp, v, w, m, nprime);
                
                thread ulong u_new[LIMBS], v_new[LIMBS];
                add_mod(u_new, u, temp, m);
                sub_mod(v_new, u, temp, m);
                
                // Store back to shared memory
                for (int k = 0; k < LIMBS; ++k) {
                    shared_data[i * LIMBS + k] = u_new[k];
                    shared_data[j * LIMBS + k] = v_new[k];
                }
            }
        }
        
        twiddle_offset += step;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write results back to global memory
    if (local_id < actual_block_size) {
        uint global_idx = block_start + local_id;
        for (int k = 0; k < LIMBS; ++k) {
            data[global_idx * LIMBS + k] = shared_data[local_id * LIMBS + k];
        }
    }
}

// -----------------------------------------------------------
// Simple Butterfly Kernel - for fallback or large FFTs
// -----------------------------------------------------------
kernel void butterfly_fft(
    device ulong*       data,           // Input/output data array
    constant uint&      n,              // Total size of FFT
    constant uint&      stage,          // Current stage (0, 1, 2, ...)
    device const ulong* twiddles,       // Twiddle factors for this stage
    constant ulong*     modulus,        // Modulus
    constant ulong&     nprime,         // Montgomery parameter
    uint tid [[thread_position_in_grid]]
) {
    uint step = 1u << stage;           // Distance between butterfly elements
    uint num_groups = n >> (stage + 1); // Number of groups in this stage
    
    if (tid >= num_groups * step) return;
    
    // Calculate butterfly indices
    uint group = tid / step;
    uint pos = tid % step;
    uint i = group * (step << 1) + pos;
    uint j = i + step;
    
    if (j >= n) return;
    
    // Load modulus
    thread ulong m[LIMBS];
    for (int k = 0; k < LIMBS; ++k) {
        m[k] = modulus[k];
    }
    
    // Load twiddle factor for this butterfly position
    thread ulong w[LIMBS];
    for (int k = 0; k < LIMBS; ++k) {
        w[k] = twiddles[pos * LIMBS + k];
    }
    
    // Load data elements
    thread ulong u[LIMBS], v[LIMBS];
    for (int k = 0; k < LIMBS; ++k) {
        u[k] = data[i * LIMBS + k];
        v[k] = data[j * LIMBS + k];
    }
    
    // Butterfly operation: 
    // temp = v * w
    // u' = u + temp
    // v' = u - temp
    thread ulong temp[LIMBS];
    mont_mul(temp, v, w, m, nprime);
    
    thread ulong u_new[LIMBS], v_new[LIMBS];
    add_mod(u_new, u, temp, m);
    sub_mod(v_new, u, temp, m);
    
    // Store results
    for (int k = 0; k < LIMBS; ++k) {
        data[i * LIMBS + k] = u_new[k];
        data[j * LIMBS + k] = v_new[k];
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