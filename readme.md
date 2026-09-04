# CUDA accelerated ZODA encoding over the BabyBear field

This is a CUDA implementation of ZODA encoding that operates over the `u32` BabyBear prime field.
The goal of this repository is to find out how large a Celestia-style data square we can encode on a
single GPU, and more importantly what actually limits us when we try.

All numbers in this document were measured on one machine, a single RTX 5090 with driver `595.84`
and CUDA toolkit `12.4`. They are reproducible with the commands below, but they describe this one
setup and not GPUs in general.

The short version of what we found: the kernel encodes at `61.7 Gb/s`, which is more than the
`34-45 Gbps` network link of a node in Celestia's Fibre cluster. The pipeline we wrap around that
kernel only delivers `6.1 Gb/s`, because `90%` of the wall clock goes into a single-threaded host
transpose. The GPU is not our bottleneck. The code on either side of it is.

## Prerequisite Knowledge
- Finite Field Arithmetic
- The Number Theoretic Transform (see [this article](https://github.com/jonas089/articles/blob/master/01-fft.md))
- Reed-Solomon erasure coding
- Basic CUDA (streams, pinned memory, kernel launches)

## Running the benchmarks

```bash
# correctness
cargo test test_zoda_babybear_gpu --features cuda --release -- --nocapture
# throughput on Celestia data squares
ZODA_BENCH_ITERS=5 cargo test benchmark_celestia_squares --features cuda --release -- --ignored --nocapture
```

## The kernel

`cuda/ntt_kernel.cu` transforms every column of a row-major `[rows][cols]` square in a single call.
It mirrors the radix-2 CPU reference in `src/ntt/cpu.rs`: we bit-reverse, then we issue one kernel
launch per stage, with one thread per butterfly per column.

Our field elements are `u32` BabyBear. On the device we keep them in Montgomery form so that a
multiplication needs no division, and on the Rust side we keep them canonical. The reason this
matters so much is that a GPU has no 64-bit integer division instruction, so a naive `% p` in the
butterfly expands into a long software sequence. With Montgomery reduction we get a multiply down to
two multiplications and a `min`.

We process columns in chunks of `1024` that rotate over three streams, so that our uploads and
downloads overlap the kernels. Note that this chunking is about overlap and not about volume, which
turns out to matter later.

## What we measure

A square of side `s` holds `s * s` shares of `512` bytes. ZODA extends it column-wise to `2s` rows,
so in our configuration `k = n = s`. Every figure below is the median of `5` timed encodes after one
untimed warm-up.

We time two windows, and the distinction between them carries the whole argument:

- **`GPU xform`** is the two transform calls: upload, INTT, NTT, download.
- **`encode`** is end-to-end, and additionally covers the host-side transpose into the pinned
  column-major buffer and the zero padding.

That means `encode - GPU xform` is exactly our single-threaded host layout work. Throughput is
payload per second of the relevant window. We report `Gb/s` in decimal gigabits so that it compares
directly against network figures. Generating the input and running the correctness check are not
timed, since a real encoder does neither.

## Results

| Square | Payload | Extended | `GPU xform` | `encode` | `GPU xform` Gb/s | `encode` Gb/s | Validated |
|--------|---------|----------|-------------|----------|------------------|---------------|-----------|
| 64x64 | 2 MiB | 4 MiB | 0.6 ms | 1.0 ms | 26.1 | 16.1 | yes |
| 128x128 | 8 MiB | 16 MiB | 1.6 ms | 5.8 ms | 42.9 | 11.6 | yes |
| 256x256 | 32 MiB | 64 MiB | 5.2 ms | 35.6 ms | 51.6 | 7.5 | yes |
| 512x512 | 128 MiB | 256 MiB | 18.9 ms | 170.8 ms | 56.8 | 6.3 | yes |
| 1024x1024 | 512 MiB | 1024 MiB | 71.0 ms | 679.9 ms | 60.5 | 6.3 | yes |
| 2048x2048 | 2048 MiB | 4096 MiB | 278.4 ms | 2831.5 ms | 61.7 | 6.1 | yes |
| 4096x4096 | 8192 MiB | 16384 MiB | 1114.1 ms | 45657.4 ms | 61.7 | 1.5 | skipped |

> [!NOTE]
> Our validation re-encodes every column on the CPU and needs a second host copy of the square, so
> we skip it above side `2048`. The table says `skipped` rather than `yes`, because we did not check
> that square and should not imply that we did.

> [!WARNING]
> The `4096x4096` `encode` figure is degraded by host memory pressure. The pinned square alone is
> `16 GiB` on a `59 GiB` box, so that number understates our pipeline and should not be used as a
> baseline. Its `GPU xform` figure is unaffected, because the device only ever holds streamed chunks.

## The kernel gets faster on bigger squares, then hits PCIe

The first thing to notice is that our transform throughput *rises* with the square side, from
`26.1 Gb/s` at `64x64` to `61.7 Gb/s` at `2048x2048`. That is a `2.4x` gain simply from growing the
square.

The reason is that larger squares give the kernel more parallel work per launch, and they amortize
our fixed per-call setup over more data. That setup is not free: we allocate device memory, create
three streams and issue `log2(n)` launches per chunk on every call. At `64x64` there is not enough
work to hide any of it. So the regime our kernel is *worst* at is the small one, and there is no
scaling wall as squares grow. That is the property we care about if the square is going to be raised
over time.

Then it flattens. We measure `61.7 Gb/s` at both `2048` and `4096`. This plateau is PCIe volume and
not the butterflies. At `2048x2048` our payload is `2.147 GB`, but `12.885 GB` crosses the bus, which
is a `6.0x` amplification. In `278.4 ms` that works out to `46.3 GB/s` of wire bandwidth, which is
about where real pinned transfers land on this card's Gen5 x16 link.

In other words, our Montgomery arithmetic is completely hidden behind the transfers. We will see
below where the extra `5x` of traffic comes from and why the same fix lifts both this number and our
end-to-end one.

### What a time budget buys us

Because our payload is `side^2 * 512`, the side that a given rate sustains grows as the *square
root* of the time budget. A `6x` budget therefore buys about `2.4x` the side and not `6x`. Note that
this is a property of squares and has nothing to do with our kernel.

| Budget | Side at `GPU xform` rate | Side at `encode` rate |
|--------|--------------------------|-----------------------|
| 1 s | 3881 | 1217 |
| 6 s | 9507 | 2981 |

> [!NOTE]
> These two columns are extrapolated from our largest *validated* square, `2048x2048`. We did not
> measure at these sides. We deliberately do not extrapolate from the peak `encode` rate, because
> that peak occurs at the smallest square and would overstate the ceiling by roughly `2.7x`.

## The ceiling is host-side, not the GPU

At `2048x2048` our transforms take `278 ms` of a `2832 ms` encode. That means `90%` of the wall clock
is the single-threaded host transpose, and GPU utilisation sat at `16%` across the whole run. We
deliver `6.1 Gb/s` end-to-end where the kernel does `61.7`, so we are throwing away a factor of `10`.

Device memory is not what caps our square size. Because columns stream through in chunks, the device
only holds `ntt_size_kn * 1024` elements per stream, which is `50 MB` of the card's `32 GB` at
`2048x2048`. So VRAM will not stop us growing the square.

> [!NOTE]
> That headroom is not spare throughput. If we enlarged the chunks we would move the same number of
> bytes across PCIe, and with only three streams we would coarsen the upload/compute/download
> overlap. In the limit of a single chunk we would lose the overlap entirely and get slower. Our
> binding constraints are host memory, host layout work and PCIe volume, not the accelerator.

Three costs sit in that gap, and all three are data flow rather than arithmetic:

1. **We build the zero padding on the host and then upload it.** At `2048x2048` half of what our
   forward NTT sends over PCIe is zeros.
2. **`intt_cuda` and `ntt_cuda` are separate calls.** Each one allocates, uploads and downloads, so
   our square makes a full round trip to host memory between the two transforms.
3. **Our transpose is single-threaded and scattered.** It writes with a stride of `ntt_size_kn`
   elements, so at large sizes nearly every write lands on a different page and the page walker
   becomes the bottleneck.

The first two are measurable as PCIe volume. At `2048x2048`:

| | Bytes over PCIe |
|---|---|
| Payload | 2.147 GB |
| intt upload + download | 4.295 GB |
| ntt upload + download (upload is half zeros) | 8.590 GB |
| **Total today** | **12.885 GB**, `6.0x` the payload |
| Fused, transpose and pad on device | **6.442 GB**, `2.0x` less |

If we upload row-major input and do the transpose and the zero fill on the device, we address all
three at once. We halve our PCIe traffic, we remove most of the host time, and we lift the transform
plateau, because that plateau *is* the traffic. At the `46.3 GB/s` of wire bandwidth we already
measure, `6.442 GB` would take about `139 ms`, which is roughly `123 Gb/s` of payload and double
today's kernel figure.

## Context: Celestia Fibre

Celestia's [Fibre announcement](https://blog.celestia.org/introducing-fibre-1tb-s-of-blockspace/)
reports `1 Tb/s` aggregate across `498` GCP machines, each with `48-64` vCPUs, `90-128 GB` of RAM and
`34-45 Gbps` network links. Note that the post describes this as a test of the *networking layer*.
It publishes no encoding throughput figure and makes no CPU-or-GPU claim about ZODA encoding. In
Fibre, users encode their own blobs and distribute the pieces to validators, so those `498` machines
propagate rather than encode a square.

This means `1 Tb/s` is not an encoding ceiling, and we cannot claim that our kernel beats it. The
two numbers measure different things. What their published per-machine specs do let us do is compare
against a single node's capacity:

| Fibre figure | Value | This kernel, one RTX 5090 |
|--------------|-------|---------------------------|
| Per-node network link | 34-45 Gbps | 61.7 Gb/s `GPU xform` |
| Max blobsize | 128 MB | 18.9 ms per 128 MiB square (512x512) |

So one GPU encodes faster than a Fibre node's link can carry. A maximum-size `128 MB` blob costs us
`18.9 ms` of GPU time, which is about `53` blobs per second.

> [!NOTE]
> The announcement's "881x faster than KZG" claim comes with no stated hardware, field or
> measurement window, so we do not build anything here on it.

### Conditional: if 2 Gb/s were a node's encode ceiling

If we divide `1 Tb/s` by `498` nodes we get roughly `2 Gb/s` per node. *If* that were a node's
encoding ceiling rather than its share of the test, then removing it would be worth:

| | Per-node gain over 2 Gb/s |
|---|---|
| This kernel, unconstrained | 61.7 Gb/s, about `31x` |
| Capped by the node's own link | 34-45 Gbps, about `17-22x` |
| Our shipped end-to-end pipeline | 6.1 Gb/s, about `3x` |

Under that premise the realistic figure is the middle row, because our kernel has headroom past the
link and so the link binds first. The bottom row is what our encoder delivers today, and getting from
there to the middle row needs the data-flow fix described above.

> [!WARNING]
> The premise is an assumption, and probably a weak one. `2 Gb/s` is `1 Tb/s / 498`, which is an
> output of the test's design and not a published per-node encode measurement. If we change the node
> count the figure changes with it, without telling us anything about what a node can do. The
> corroborating detail points the same way: those machines had `34-45 Gbps` links and carried about
> `2 Gb/s`, roughly `5%` utilisation. That is a strange amount of network to provision if encoding
> capped them `20x` below it. The likelier reading is that node count was chosen to reach `1 Tb/s`
> aggregate.

The claim that does not need this premise is the capacity comparison above. `61.7 Gb/s` of encode
against a `34-45 Gbps` link means encoding is not what a node has to be provisioned around.

## What we would optimize next

Ordered by how much we expect each step to buy us:

1. **Upload row-major and transpose on the device.** This removes the `90%` host cost and stops us
   shipping host-built zeros. It is the single biggest win available.
2. **Fuse the INTT and the NTT into one call.** Right now the square round-trips to host memory
   between them for no reason other than that they are two independent `run()` calls that each own
   their allocation.
3. **Block the butterfly stages in shared memory.** We currently issue one launch per stage, so the
   whole array round-trips through DRAM once per stage. A blocked NTT would cut that to a handful of
   passes. Note that this is invisible today because PCIe hides it, and it only starts to matter once
   steps 1 and 2 are done or once the square stays resident on the device.
4. **Parallelize the transpose** if we want a contained interim fix without touching the kernel. It
   is memory-bound and trivially parallel by column block.

## Correctness

Every square up to side `2048` in the results table is validated against the CPU reference.
`src/benchmarks/zoda_validated.rs` additionally checks full ZODA soundness in two phases:

1. **Column encoding.** Every sampled column must be a valid Reed-Solomon codeword and must match
   the CPU reference exactly.
2. **RLC soundness.** We derive coefficients from the commitment, extend each row's random linear
   combination from `k` to `k+n`, and check the extended rows for consistency.

Together these give us column-wise codeword validity, row-wise linear-combination consistency and
commitment binding against forgery.

```bash
cargo test --features cuda --release benchmark_zoda_validated -- --ignored --nocapture
```

## Conclusion

1. **A single GPU already exceeds a Fibre node's network link for encoding.** `61.7 Gb/s` of
   transform against a `34-45 Gbps` link means encode compute is not what limits a node.
2. **Bigger squares suit our kernel better, not worse.** Throughput rises `2.4x` from side `64` to
   `2048` and then holds flat. Nothing in the measured range degrades as the square grows.
3. **Both remaining limits are data flow, and one change fixes both.** Our kernel plateau is PCIe
   volume, caused by shipping host-built zeros and round-tripping the square between the two
   transforms. Our end-to-end gap is a single-threaded scattered transpose. Uploading row-major input
   and doing the transpose and zero fill on the device removes the host time *and* halves the wire
   traffic.

Read against Fibre's per-node figures, the defensible statement is that our encode capacity exceeds
a node's network link. A larger per-node multiple of about `17-22x` follows only if we assume
`2 Gb/s` is a node's encode ceiling, which the published material does not establish.

So neither the field arithmetic nor the GPU is our constraint on how large a data square we can
encode per block. Byte movement is, on the bus and through host memory, and that is a data-flow
problem rather than a faster-butterfly problem.