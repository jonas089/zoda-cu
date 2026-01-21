/*use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Check if CUDA is available by trying to run nvcc
    let nvcc_available = Command::new("nvcc")
        .arg("--version")
        .output()
        .is_ok();

    if !nvcc_available {
        println!("cargo:warning=nvcc not found, CUDA support will be disabled");
        println!("cargo:rustc-cfg=feature=\"no_cuda\"");
        return;
    }

    println!("cargo:rerun-if-changed=cuda/ntt_kernel.cu");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cuda_src = "cuda/ntt_kernel.cu";
    let cuda_obj = out_dir.join("ntt_kernel.o");
    let cuda_lib = out_dir.join("libntt_cuda.a");

    // Detect CUDA compute capability
    // Default to compute_75 (RTX 20xx series) and compute_89 (RTX 40xx/50xx)
    // This should cover RTX 3060 (compute_86) and RTX 5090 (compute_89)
    let compute_capabilities = vec![
        "compute_75", // RTX 2060, 2070, 2080
        "compute_86", // RTX 3060, 3070, 3080, 3090
        "compute_89", // RTX 4090, RTX 5090
    ];

    let mut gencode_flags = Vec::new();
    for cc in &compute_capabilities {
        let sm = cc.replace("compute_", "sm_");
        gencode_flags.push(format!("-gencode"));
        gencode_flags.push(format!("arch={},code={}", cc, sm));
    }

    // Add final compute capability for forward compatibility
    gencode_flags.push(format!("-gencode"));
    gencode_flags.push(format!("arch=compute_89,code=compute_89"));

    // Compile CUDA source to object file
    let mut nvcc_cmd = Command::new("nvcc");
    nvcc_cmd
        .arg("-c")
        .arg(cuda_src)
        .arg("-o")
        .arg(&cuda_obj)
        .arg("--compiler-options")
        .arg("-fPIC")
        .arg("-O3")
        .args(&gencode_flags);

    println!("Running nvcc: {:?}", nvcc_cmd);

    let output = nvcc_cmd.output().expect("Failed to run nvcc");

    if !output.status.success() {
        eprintln!("nvcc stderr: {}", String::from_utf8_lossy(&output.stderr));
        panic!("Failed to compile CUDA code");
    }

    // Create static library from object file
    let ar_status = Command::new("ar")
        .arg("rcs")
        .arg(&cuda_lib)
        .arg(&cuda_obj)
        .status()
        .expect("Failed to run ar");

    if !ar_status.success() {
        panic!("Failed to create static library");
    }

    // Find CUDA library directory
    let cuda_lib_paths = if let Ok(cuda_path) = env::var("CUDA_PATH") {
        vec![
            format!("{}/lib64", cuda_path),
            format!("{}/lib", cuda_path),
        ]
    } else {
        vec![
            "/usr/local/cuda/lib64".to_string(),
            "/usr/local/cuda/lib".to_string(),
            "/usr/lib/x86_64-linux-gnu".to_string(),
            "/usr/lib64".to_string(),
            "/opt/cuda/lib64".to_string(),
            "/opt/cuda/lib".to_string(),
            "/usr/lib/wsl/lib".to_string(), // WSL2
        ]
    };

    // Tell cargo to link the CUDA runtime and our library
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=ntt_cuda");

    // Add all CUDA library paths
    for path in &cuda_lib_paths {
        println!("cargo:rustc-link-search=native={}", path);
    }

    // Try to link cudart dynamically first, fall back to static if needed
    println!("cargo:rustc-link-lib=dylib=cudart");

    println!("cargo:rustc-cfg=feature=\"cuda\"");
    println!("cargo:warning=CUDA support enabled");
}
*/

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Check if CUDA is available by trying to run nvcc
    let nvcc_available = Command::new("nvcc").arg("--version").output().is_ok();

    if !nvcc_available {
        println!("cargo:warning=nvcc not found, CUDA support will be disabled");
        println!("cargo:rustc-cfg=feature=\"no_cuda\"");
        return;
    }

    println!("cargo:rerun-if-changed=cuda/ntt_kernel.cu");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cuda_src = "cuda/ntt_kernel.cu";
    let cuda_obj = out_dir.join("ntt_kernel.o");
    let cuda_lib = out_dir.join("libntt_cuda.a");

    // Detect CUDA compute capability
    // Default to compute_75 (RTX 20xx series) and compute_89 (RTX 40xx/50xx)
    // This should cover RTX 3060 (compute_86) and RTX 5090 (compute_89)
    let compute_capabilities = vec![
        "compute_75", // RTX 2060, 2070, 2080
        "compute_86", // RTX 3060, 3070, 3080, 3090
        "compute_89", // RTX 4090, RTX 5090
    ];

    let mut gencode_flags = Vec::new();
    for cc in &compute_capabilities {
        let sm = cc.replace("compute_", "sm_");
        gencode_flags.push(format!("-gencode"));
        gencode_flags.push(format!("arch={},code={}", cc, sm));
    }

    // Add final compute capability for forward compatibility
    gencode_flags.push(format!("-gencode"));
    gencode_flags.push(format!("arch=compute_89,code=compute_89"));

    // Compile CUDA source to object file
    let mut nvcc_cmd = Command::new("nvcc");
    nvcc_cmd
        .arg("-c")
        .arg(cuda_src)
        .arg("-o")
        .arg(&cuda_obj)
        .arg("--compiler-options")
        .arg("-fPIC")
        .arg("-O3")
        .args(&gencode_flags);

    println!("Running nvcc: {:?}", nvcc_cmd);

    let output = nvcc_cmd.output().expect("Failed to run nvcc");

    if !output.status.success() {
        eprintln!("nvcc stderr: {}", String::from_utf8_lossy(&output.stderr));
        panic!("Failed to compile CUDA code");
    }

    // Create static library from object file
    let ar_status = Command::new("ar")
        .arg("rcs")
        .arg(&cuda_lib)
        .arg(&cuda_obj)
        .status()
        .expect("Failed to run ar");

    if !ar_status.success() {
        panic!("Failed to create static library");
    }

    // Find CUDA library directory
    let cuda_lib_paths = if let Ok(cuda_path) = env::var("CUDA_PATH") {
        vec![format!("{}/lib64", cuda_path), format!("{}/lib", cuda_path)]
    } else {
        vec![
            "/usr/local/cuda/lib64".to_string(),
            "/usr/local/cuda/lib".to_string(),
            "/usr/lib/x86_64-linux-gnu".to_string(),
            "/usr/lib64".to_string(),
            "/opt/cuda/lib64".to_string(),
            "/opt/cuda/lib".to_string(),
            "/usr/lib/wsl/lib".to_string(), // WSL2
        ]
    };

    // Tell cargo to link the CUDA runtime and our library
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=ntt_cuda");

    // Add all CUDA library paths
    for path in &cuda_lib_paths {
        println!("cargo:rustc-link-search=native={}", path);
    }

    // Try to link cudart dynamically first, fall back to static if needed
    println!("cargo:rustc-link-lib=dylib=cudart");

    println!("cargo:rustc-cfg=feature=\"cuda\"");
    println!("cargo:warning=CUDA support enabled");
}
