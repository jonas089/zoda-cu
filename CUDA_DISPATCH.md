# CUDA Kernel Dispatch and Result Collection

This document shows the **exact code paths** where CUDA kernels are dispatched and results are collected.

## Overview: Two Main Entry Points

1. **`benchmark_zoda_optimal.rs`** - Pure performance benchmark
2. **`benchmark_zoda_validated.rs`** - Correctness validation + benchmark

Both follow the same pattern but with different post-processing.

---

## Entry Point 1: Performance Benchmark

**File:** [src/benchmark_zoda_optimal.rs](src/benchmark_zoda_optimal.rs#L122-L183)

### Function: `encode_gpu_optimized()`

This is the **main GPU dispatch function** for benchmarking.

```rust
#[cfg(feature = "cuda")]
fn encode_gpu_optimized(config: &EncodingConfig) -> Result<f64, String> {
    let k = config.k;
    let n = config.n;
    let num_positions = config.num_positions();  // Number of columns
    let ntt_size_k = config.ntt_size_k();        // Power of 2 >= k
    let ntt_size_kn = config.ntt_size_kn();      // Power of 2 >= k+n

    // Get roots of unity for the two domains
    let omega_k = BabyBear::get_root_of_unity(ntt_size_k.trailing_zeros());
    let omega_kn = BabyBear::get_root_of_unity(ntt_size_kn.trailing_zeros());

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // STEP 1: Prepare host input (column-major layout)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    let mut h_input: Vec<u64> = vec![0; num_positions * ntt_size_k];

    // Fill input: [col_0][col_1]...[col_n-1]
    // Each column has k real values + (ntt_size_k - k) zeros
    for row_idx in 0..k {
        for col in 0..num_positions {
            let value = ((row_idx * num_positions + col) % 2013265921) as u64;
            h_input[col * ntt_size_k + row_idx] = value;
        }
    }
    // Padding from k to ntt_size_k is already zero (vec initialization)

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // STEP 2: Allocate GPU buffers
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    let mut d_input = CudaBuffer::new(num_positions * ntt_size_k)?;   // Input
    let mut d_output = CudaBuffer::new(num_positions * ntt_size_kn)?; // Output
    let mut d_work = CudaBuffer::new(num_positions * ntt_size_k)?;    // Workspace

    let mut h_output: Vec<u64> = vec![0; num_positions * ntt_size_kn];

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // STEP 3: Start timing
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    let start = Instant::now();

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // STEP 4: Upload data to GPU (H2D transfer)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    d_input.copy_from_host(&h_input)?;
    // ↑ Blocking: CPU waits for transfer to complete
    // ↑ Internal: cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice)

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // STEP 5: DISPATCH CUDA KERNEL (The Main Event!)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    unsafe {
        cuda_rs_encode_vertical(
            d_input.as_ptr(),       // Input pointer (device memory)
            d_output.as_ptr(),      // Output pointer (device memory)
            d_work.as_ptr(),        // Workspace pointer (device memory)
            num_positions as u32,   // Number of columns to process
            ntt_size_k as u32,      // Domain size for INTT
            ntt_size_kn as u32,     // Domain size for NTT
            omega_k.value,          // Root of unity for k
            omega_kn.value,         // Root of unity for k+n
        );
    }
    // ↑ This is an FFI call to C function in ntt_kernel.cu
    // ↑ The function internally:
    //   1. Copies d_input → d_work (device-to-device)
    //   2. Launches batched INTT kernel on d_work
    //   3. Launches GPU padding kernel (d_work → d_output)
    //   4. Launches batched NTT kernel on d_output
    //   5. Calls cudaDeviceSynchronize() to wait for all kernels
    // ↑ When this returns, GPU work is COMPLETE

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // STEP 6: Download results from GPU (D2H transfer)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    d_output.copy_to_host(&mut h_output)?;
    // ↑ Blocking: CPU waits for transfer to complete
    // ↑ Internal: cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost)

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // STEP 7: Stop timing
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    // Output layout: h_output[col * ntt_size_kn + row]
    // Where: col ∈ [0, num_positions), row ∈ [0, k+n)

    Ok(elapsed)
}
```

### What Happens Inside `cuda_rs_encode_vertical()`

**File:** [cuda/ntt_kernel.cu:411-447](cuda/ntt_kernel.cu#L411-L447)

```c
void cuda_rs_encode_vertical(
    const uint64_t* d_input,     // GPU pointer
    uint64_t* d_output,          // GPU pointer
    uint64_t* d_intt_work,       // GPU pointer
    uint32_t num_positions,      // 4096 (for 1GB benchmark)
    uint32_t ntt_size_k,         // 65536
    uint32_t ntt_size_kn,        // 131072
    uint64_t omega_k,            // Root of unity
    uint64_t omega_kn            // Root of unity
) {
    uint32_t threads = 256;

    // Internal Step 1: Copy input to workspace (D2D, fast!)
    cudaMemcpy(d_intt_work, d_input,
               num_positions * ntt_size_k * sizeof(uint64_t),
               cudaMemcpyDeviceToDevice);

    // Internal Step 2: Batched INTT (ALL columns in parallel)
    cuda_intt_batched(d_intt_work, num_positions, ntt_size_k,
                      ntt_size_k, omega_k);
    // ↑ Processes 4096 INTTs simultaneously
    // ↑ Each INTT transforms k evaluation points → coefficients

    // Internal Step 3: GPU padding (k → k+n with zeros)
    uint32_t total_padded = num_positions * ntt_size_kn;
    uint32_t pad_blocks = (total_padded + threads - 1) / threads;
    gpu_pad_batched<<<pad_blocks, threads>>>(
        d_intt_work, d_output,
        num_positions, ntt_size_k, ntt_size_kn
    );
    // ↑ Zero-pad polynomial coefficients
    // ↑ NO CPU ROUNDTRIP! Stays on GPU.

    // Internal Step 4: Batched NTT (ALL columns in parallel)
    cuda_ntt_batched(d_output, num_positions, ntt_size_kn,
                     ntt_size_kn, omega_kn);
    // ↑ Processes 4096 NTTs simultaneously
    // ↑ Each NTT evaluates polynomial at k+n points

    // Internal Step 5: Wait for all GPU work to finish
    cudaDeviceSynchronize();
    // ↑ Blocks until all kernels complete
    // ↑ After this, d_output contains final result
}
```

---

## Entry Point 2: Validated Benchmark

**File:** [src/benchmark_zoda_validated.rs](src/benchmark_zoda_validated.rs#L51-L112)

### Function: `encode_gpu_with_output()`

This is nearly identical to `encode_gpu_optimized()`, but includes **row-major conversion** for validation.

```rust
#[cfg(feature = "cuda")]
fn encode_gpu_with_output(config: &EncodingConfig)
    -> Result<(Vec<Vec<BabyBear>>, f64), String> {

    let k = config.k;
    let n = config.n;
    let num_positions = config.num_positions();
    let ntt_size_k = config.ntt_size_k();
    let ntt_size_kn = config.ntt_size_kn();

    let omega_k = BabyBear::get_root_of_unity(ntt_size_k.trailing_zeros());
    let omega_kn = BabyBear::get_root_of_unity(ntt_size_kn.trailing_zeros());

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // STEP 1: Prepare host input (column-major)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    let mut h_input: Vec<u64> = Vec::with_capacity(
        num_positions * ntt_size_k
    );

    for col in 0..num_positions {
        for row in 0..k {
            let value = ((row * num_positions + col) % 2013265921) as u64;
            h_input.push(value);
        }
        for _ in k..ntt_size_k {
            h_input.push(0);  // Zero padding
        }
    }

    let mut h_output = vec![0u64; num_positions * ntt_size_kn];

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // STEP 2: Allocate GPU buffers
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    let mut d_input = CudaBuffer::new(total_input_size)?;
    let mut d_output = CudaBuffer::new(total_output_size)?;
    let d_work = CudaBuffer::new(work_size)?;

    let start = Instant::now();

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // STEP 3: Upload (H2D)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    d_input.copy_from_host(&h_input)?;

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // STEP 4: DISPATCH CUDA KERNEL
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    unsafe {
        cuda_rs_encode_vertical(
            d_input.as_ptr(),
            d_output.as_ptr(),
            d_work.as_ptr(),
            num_positions as u32,
            ntt_size_k as u32,
            ntt_size_kn as u32,
            omega_k.value,
            omega_kn.value,
        );
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // STEP 5: Download (D2H)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    d_output.copy_to_host(&mut h_output)?;

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // STEP 6: Convert column-major → row-major for validation
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    let mut encoded_rows: Vec<Vec<BabyBear>> =
        vec![vec![BabyBear::zero(); num_positions]; k + n];

    for col in 0..num_positions {
        for row in 0..(k + n) {
            // Read from column-major: h_output[col * stride + row]
            encoded_rows[row][col] = BabyBear::new(
                h_output[col * ntt_size_kn + row]
            );
        }
    }

    Ok((encoded_rows, elapsed))
}
```

### Key Difference: Row-Major Conversion

The validated benchmark needs **row-major format** for validation:

```rust
// GPU output is column-major: [col_0][col_1]...[col_n-1]
// Each column has k+n values

// Convert to row-major: rows[row_idx][col_idx]
for col in 0..num_positions {
    for row in 0..(k + n) {
        encoded_rows[row][col] = BabyBear::new(
            h_output[col * ntt_size_kn + row]
            //        ↑ column base     ↑ offset within column
        );
    }
}

// Result: encoded_rows[row][col]
// This format is convenient for:
//   - Extracting entire rows
//   - Computing RLC per row
//   - Comparing against CPU row-by-row
```

---

## Memory Layout Deep Dive

### Before Dispatch (Host Input)

```
h_input layout (column-major):
┌─────────────────────────────────────┐
│ Column 0: [val_0, val_1, ..., 0]   │  ntt_size_k elements
│ Column 1: [val_0, val_1, ..., 0]   │  (k real + padding)
│ Column 2: [val_0, val_1, ..., 0]   │
│ ...                                 │
│ Column m-1: [val_0, val_1, ..., 0] │
└─────────────────────────────────────┘
Total: num_positions × ntt_size_k u64s

Access pattern:
h_input[col * ntt_size_k + row]
```

### After Collection (Host Output)

```
h_output layout (column-major):
┌─────────────────────────────────────┐
│ Column 0: [val_0, ..., val_(k+n-1)]│  ntt_size_kn elements
│ Column 1: [val_0, ..., val_(k+n-1)]│
│ Column 2: [val_0, ..., val_(k+n-1)]│
│ ...                                 │
│ Column m-1: [val_0, ..., val_kn]   │
└─────────────────────────────────────┘
Total: num_positions × ntt_size_kn u64s

Access pattern:
h_output[col * ntt_size_kn + row]
```

### Row-Major Conversion (for validation)

```
encoded_rows[row][col] ← h_output[col * ntt_size_kn + row]

Result:
┌───────────────────────────────────────┐
│ Row 0: [col_0_val, col_1_val, ...]   │
│ Row 1: [col_0_val, col_1_val, ...]   │
│ ...                                   │
│ Row k+n-1: [col_0_val, ...]          │
└───────────────────────────────────────┘
Total: (k+n) rows × num_positions BabyBear
```

---

## FFI Boundary: Rust → CUDA

### Rust Side: `cuda_ntt.rs`

```rust
// FFI declaration
#[link(name = "ntt_cuda", kind = "static")]
extern "C" {
    pub fn cuda_rs_encode_vertical(
        d_input: *const u64,
        d_output: *mut u64,
        d_intt_work: *mut u64,
        num_positions: u32,
        ntt_size_k: u32,
        ntt_size_kn: u32,
        omega_k: u64,
        omega_kn: u64,
    );
}
```

**Calling convention:**
- `extern "C"` ensures C ABI compatibility
- Pointers are raw device pointers (from `CudaBuffer::as_ptr()`)
- No Rust ownership transfer across FFI boundary
- Function is `unsafe` because:
  1. Raw pointers
  2. GPU synchronization contract
  3. No Rust memory safety guarantees on device memory

### CUDA Side: `ntt_kernel.cu`

```c
extern "C" {
    void cuda_rs_encode_vertical(
        const uint64_t* d_input,
        uint64_t* d_output,
        uint64_t* d_intt_work,
        uint32_t num_positions,
        uint32_t ntt_size_k,
        uint32_t ntt_size_kn,
        uint64_t omega_k,
        uint64_t omega_kn
    ) {
        // Implementation here...
        cudaDeviceSynchronize();  // CRITICAL: must sync before return
    }
}
```

**Contract:**
- Must call `cudaDeviceSynchronize()` before returning
- All GPU work must be complete when function returns
- No async behavior from Rust's perspective
- Device pointers remain valid (managed by `CudaBuffer`)

---

## RAII Memory Management

### CudaBuffer Lifecycle

```rust
pub struct CudaBuffer {
    ptr: *mut u64,
    size: usize,
}

impl CudaBuffer {
    pub fn new(size: usize) -> Result<Self, String> {
        let mut ptr: *mut u64 = std::ptr::null_mut();
        unsafe {
            let err = cuda_malloc(&mut ptr, size);
            if err != CUDA_SUCCESS {
                return Err("CUDA malloc failed");
            }
        }
        Ok(Self { ptr, size })
    }

    pub fn as_ptr(&self) -> *mut u64 {
        self.ptr
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        unsafe {
            cuda_free(self.ptr);  // Always called when going out of scope
        }
    }
}
```

**Ownership flow:**

```rust
{
    // Allocate
    let d_input = CudaBuffer::new(size)?;
    let d_output = CudaBuffer::new(size)?;
    let d_work = CudaBuffer::new(size)?;

    // Use
    d_input.copy_from_host(&data)?;
    unsafe { cuda_rs_encode_vertical(...); }
    d_output.copy_to_host(&mut result)?;

    // Automatic cleanup on scope exit
    // Drop is called for d_input, d_output, d_work
    // GPU memory is freed even if there was an error!
}
```

**Error handling:**

```rust
fn encode_gpu_optimized() -> Result<f64, String> {
    let d_input = CudaBuffer::new(size)?;  // If this fails, return early
    let d_output = CudaBuffer::new(size)?; // If this fails, d_input is dropped
    let d_work = CudaBuffer::new(size)?;   // If this fails, d_input+d_output dropped

    d_input.copy_from_host()?;             // If this fails, all buffers dropped
    // ...

    Ok(elapsed)
    // Success: all buffers dropped here
}
```

No memory leaks possible! RAII ensures cleanup.

---

## Timing Breakdown

### What's Included in Benchmarks

```rust
let start = Instant::now();

d_input.copy_from_host(&h_input)?;        // H2D transfer (included)

unsafe {
    cuda_rs_encode_vertical(              // GPU compute (included)
        d_input.as_ptr(),
        d_output.as_ptr(),
        d_work.as_ptr(),
        num_positions as u32,
        ntt_size_k as u32,
        ntt_size_kn as u32,
        omega_k.value,
        omega_kn.value,
    );
}

d_output.copy_to_host(&mut h_output)?;    // D2H transfer (included)

let elapsed = start.elapsed().as_secs_f64() * 1000.0;  // Total time
```

**Timing includes:**
- ✅ H2D transfer (upload)
- ✅ GPU kernel execution (INTT + pad + NTT)
- ✅ D2H transfer (download)
- ✅ CUDA synchronization overhead

**Timing excludes:**
- ❌ Host data preparation (`h_input` creation)
- ❌ Buffer allocation (`CudaBuffer::new()`)
- ❌ Row-major conversion (if any)
- ❌ Validation logic

---

## Summary: The Complete Path

```
┌──────────────────────────────────────────────────────────────┐
│ Rust: encode_gpu_optimized() or encode_gpu_with_output()    │
└────────────────────┬─────────────────────────────────────────┘
                     │
     ┌───────────────▼────────────────┐
     │ 1. Prepare h_input (column-major)│
     └───────────────┬────────────────┘
                     │
     ┌───────────────▼────────────────┐
     │ 2. Allocate CudaBuffer x3      │
     │    (d_input, d_output, d_work) │
     └───────────────┬────────────────┘
                     │
     ┌───────────────▼────────────────┐
     │ 3. d_input.copy_from_host()    │ ← H2D transfer
     │    (cudaMemcpy blocking)       │
     └───────────────┬────────────────┘
                     │
     ┌───────────────▼─────────────────────────┐
     │ 4. unsafe { cuda_rs_encode_vertical() } │ ← FFI call
     └───────────────┬─────────────────────────┘
                     │
    ┌────────────────▼───────────────┐
    │ C: cuda_rs_encode_vertical()   │
    │                                │
    │  • cudaMemcpy D2D (input→work)│
    │  • cuda_intt_batched<<<>>>()   │ ← Kernel 1
    │  • gpu_pad_batched<<<>>>()     │ ← Kernel 2
    │  • cuda_ntt_batched<<<>>>()    │ ← Kernel 3
    │  • cudaDeviceSynchronize()     │ ← Wait for GPU
    └────────────────┬───────────────┘
                     │
     ┌───────────────▼────────────────┐
     │ 5. d_output.copy_to_host()     │ ← D2H transfer
     │    (cudaMemcpy blocking)       │
     └───────────────┬────────────────┘
                     │
     ┌───────────────▼────────────────┐
     │ 6. Convert to row-major        │ (validation only)
     │    OR just return timing       │ (benchmark only)
     └────────────────────────────────┘
```

**Key takeaways:**
1. **Single upload, single download** - minimal PCIe traffic
2. **All compute on GPU** - INTT, pad, NTT fused together
3. **Automatic memory management** - RAII prevents leaks
4. **Synchronous from Rust perspective** - kernels complete before return
5. **Type-safe interface** - despite unsafe FFI boundary