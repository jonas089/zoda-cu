use std::env;
use std::path::PathBuf;
use std::process::Command;

fn has_nvcc() -> bool {
    Command::new("nvcc")
        .arg("--version")
        .output()
        .is_ok()
}

fn main() {
    // Only build CUDA if:
    //   1) Cargo feature "cuda" is enabled
    //   2) nvcc is actually available
    let want_cuda = env::var("CARGO_FEATURE_CUDA").is_ok();

    if !want_cuda {
        println!("cargo:warning=CUDA feature not enabled");
        return;
    }

    if !has_nvcc() {
        println!("cargo:warning=nvcc not found — CUDA disabled");
        return;
    }

    println!("cargo:warning=CUDA enabled");

    println!("cargo:rerun-if-changed=cuda/ntt_kernel.cu");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cuda_src = "cuda/ntt_kernel.cu";
    let cuda_obj = out_dir.join("ntt_kernel.o");
    let cuda_lib = out_dir.join("libntt_cuda.a");

    // Compute capabilities — safe defaults
    let compute_caps = [
        "compute_75", // RTX 20xx
        "compute_86", // RTX 30xx
        "compute_89", // RTX 40xx / 50xx
    ];

    let mut gencode = Vec::new();
    for cc in &compute_caps {
        let sm = cc.replace("compute_", "sm_");
        gencode.push("-gencode".into());
        gencode.push(format!("arch={},code={}", cc, sm));
    }

    gencode.push("-gencode".into());
    gencode.push("arch=compute_89,code=compute_89".into());

    // Compile CUDA
    let status = Command::new("nvcc")
        .arg("-c")
        .arg(cuda_src)
        .arg("-o")
        .arg(&cuda_obj)
        .arg("-Xcompiler")
        .arg("-fPIC")
        .arg("-O3")
        .args(&gencode)
        .status()
        .expect("Failed to spawn nvcc");

    if !status.success() {
        panic!("nvcc failed");
    }

    // Archive into static library
    let status = Command::new("ar")
        .arg("rcs")
        .arg(&cuda_lib)
        .arg(&cuda_obj)
        .status()
        .expect("Failed to spawn ar");

    if !status.success() {
        panic!("ar failed");
    }

    // Tell cargo where to find libntt_cuda.a
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=ntt_cuda");

    // CUDA runtime search paths
    let cuda_lib_paths = [
        "/usr/local/cuda/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64",
    ];

    for path in cuda_lib_paths {
        println!("cargo:rustc-link-search=native={}", path);
    }

    // Link CUDA runtime
    println!("cargo:rustc-link-lib=dylib=cudart");

    // Required for CUDA C++
    println!("cargo:rustc-link-lib=stdc++");

    // Tell Rust code that CUDA is really available
    println!("cargo:rustc-cfg=has_cuda");
}
