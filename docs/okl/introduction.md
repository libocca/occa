# Introduction

While OCCA aims to abstract device management, OKL (like mon-*ocle*) aims to abstract the programming of devices.

In a nutshell, OKL extends the C language by using `@attributes` for code transformations and applies restrictions for exposing loop-based parallelism.

# Loop Parallelism

The programming model found in OKL is heavily based on

- OpenMP's parallel loop structure
- GPGPU programming model found in CUDA and OpenCL

For example, the most basic OpenMP code looks similar to

```cpp
#pragma omp parallel
for (int i = 0; i < N; ++i) {
  // Parallel work
}
```

Similarly, if we were to program this in CUDA or OpenCL it would look like

```cpp
__global__ void loop(const int N, ...) {
   const int id = threadIdx.x + (blockIdx.x * blockDim.x);
   if (id < N) {
     // Parallel work
   }
}

loop<<<gridSize, blockSize>>>(N, ...);
```

The analogous OKL code for both cases would look like

```cpp
@kernel void loop(const int N, ...) {
  for (int group = 0; group < N; group += blockSize; @outer) {
    for (int id = group; id < (group + blockSize); ++id; @inner) {
      if (id < N) {
        // Parallel work
      }
    }
  }
}
```

Two things to note in the above code snippet include:

- The use of `@kernel` to state the `loop` function is actually a kernel
- The for-loops contain a 4th statement with `@outer` and `@inner` attributes

We use annotations to explicitly expose loop-parallelism and different levels of parallel granularity

- **CUDA:**
<div style="width: 17px; display: inline-block"></div>
block / thread
- **OpenCL:**
<div style="width: 2px; display: inline-block"></div>
workgroup / workitem
- **OKL:**
<div style="width: 26px; display: inline-block"></div>
`@outer` / `@inner`