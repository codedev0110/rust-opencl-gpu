use ocl::{ProQue, Result};

// dims = G*L (g*num_threads + l)
//           OpenCL      CUDA      HIP
// G cores  (get_group_id, blockIdx,  __ockl_get_group_id, threadgroup_position_in_grid)
// L threads (get_local_id, threadIdx, __ockl_get_local_id, thread_position_in_threadgroup)

// 128 cores, 1 thread each
// 64 cores, 2 threads each

// On NVIDIA, cores are streaming multiprocessors.
//   AD102 (4090) has 144 SMs with 128 threads each
// On AMD, cores are compute units.
//   7900XTX has 96 CUs with with 64 threads each

// GPUs have warps. Warps are groups of threads, and all modern GPUs have them as 32 threads.

// SIMD - Single Instruction Multiple Data
//   vector registers
//   float<32> (1024 bits)
//   c = a + b (on vector registers, this is a single add instruction on 32 pieces of data)

// SIMT - Single Instruction Multiple Thread
//   similar to SIMD, but load/stores are different
//   load stores are implicit scatter gather, whereas on SIMD it's explicit

//SIMD(AVX2/FMA) --> 256 bit vector registers, 8 floats per register
//SIMD(AVX-512) --> 512 bit vector registers, 16 floats per register --->CPU backends can use this to speed up computations, but it's not as efficient as GPU SIMT

//SIMT (SIMD on GPU) --> 32 threads per warp, each thread has its own registers, but they execute the same instruction on different data
//SIMT(WARP,threads) --->GPU backend 

//CPU optimization:vector (AVX2) + cache reuse
//GPU optimization:threads + memory coalescing + shared memory

// Haswell CPU: 2 FMA units per core → each can start 1 FMA every 2 cycles (throughput = 0.5 cycles)
// Combined → 1 FMA per cycle per core
// Each AVX2 FMA processes 8 floats → 8 mul + 8 add = 16 FLOPs
// => Peak: 16 FLOPs per cycle per core (if pipeline is fully utilized)



fn main() -> Result<()> {
    // Kernel source code (OpenCL C)
    //  //__global const float* a,
    //  //__global const float* b,
    let kernel_src = r#"
        __kernel void add(
            __global float* c
        ) {
            c[get_global_id(0)] = get_local_id(0);
        }
    "#;

    // Initialize ProQue (Program, Queue, Context)
    let proque = ProQue::builder()
        .src(kernel_src)
        .dims(128) // Work size (global_work_size)
        .build()?;

    // Input data
    let a_data = vec![1.0f32; 128];
    let b_data = vec![2.0f32; 128];

    // Create buffers
    let a_buffer = proque.create_buffer::<f32>()?;
    let b_buffer = proque.create_buffer::<f32>()?;
    let c_buffer = proque.create_buffer::<f32>()?;

    // Write data to device buffers
    a_buffer.cmd().write(&a_data).enq()?;
    b_buffer.cmd().write(&b_data).enq()?;

    // Build kernel and set arguments
    let kernel = proque.kernel_builder("add")
        .arg(&c_buffer)
        .build()?;
        //.arg(&a_buffer)
        //.arg(&b_buffer)

    // Execute the kernel (unsafe due to GPU execution)
    unsafe { kernel.cmd().local_work_size(4).enq()?; }

    // Read result back
    let mut c_data = vec![0.0f32; 128];
    c_buffer.cmd().read(&mut c_data).enq()?;

    // Verify output
    let mut i = 0;
    for &c in &c_data {
        if i % 16 == 0 && i != 0 {
            println!("");
        }
        i += 1;
        print!("{:>3} ", c);
        //assert_eq!(c, 3.0f32); // 1.0 + 2.0 = 3.0
    }
    println!("");

    Ok(())
}