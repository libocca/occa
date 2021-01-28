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

```okl
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

- The use of `@kernel`, labeling functions that will be called by `occa::kernel` objects.
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

?> Checkout the [Loops in Depth](/guide/okl/loops-in-depth) section for more details about outer and inner loops

## Outer Loops

Outer loops (for-loops tagged with an `@outer` attribute) make the assumption that all work inside can be completely parallelized.
For example:

```okl
for (int i = 0; i < 3; ++i; @outer) {
  work(i);
}
```

should be expected to work regardless of the iteration ordering, such as:

```cpp
work(0);
work(1);
work(2);
```

and also

```cpp
work(2);
work(0);
work(1);
```

Since the execution order can be non-deterministic and in parallel, there shouldn't be any loop dependencies.
For example, this is **not** allowed:

```okl
for (int i = 0; i < 3; ++i; @outer) {
  A[i + 1] = A[i];
}
```

## Inner Loops

Similar to outer loops, inner loops (for-loops tagged with an `@inner` attribute) are expected to work when parallelized and run out-of-order.

The main difference between inner and outer loops is the ability to synchronize between inner loops.

For example, this is valid:

```okl
for (...; @outer) {
  for (int i = 0; i < 5; ++i; @inner) {
    A[i] = i;
  }
  for (int i = 0; i < 5; ++i; @inner) {
    B[i] = A[4 - i];
  }
}
```

?>
    Distributing work between outer and inner loops is heavily dependent on the device architecture.
    Try aiming for a power of 2 size for inner-loop sizes to make use of vectorization.

# Memory Spaces

There are two special _memory spaces_ introduced in OKL:

- `@shared`
- `@exclusive`

## Shared Memory

The concept of _shared_ memory is taken from the GPU programming model, where parallel threads/workitems can share data.

Adding the `@shared` attribute when declaring a variable type will allow the data to be shared across inner loop iterations.

?>
    Shared memory array sizes must be known at kernel compile time.
    However, we can specify the size at the application's runtime thanks to our JIT kernel compilation.

A super simple and non-efficient example, we could use shared memory for a prefix-sum:

```okl
for (...; @outer) {
  @shared int s_A[4];
  for (int i = 0; i < 4; ++i; @inner) {
    s_A[i] = A[i];
    A[i] = 0;
  }
  for (int i = 0; i < 4; ++i; @inner) {
    for (int j = 0; j < (i - 1); ++j) {
      A[i] += s_A[j];
    }
  }
}
```

Common uses of shared memory include:

- Parallel fetching of data
- Reuse of fetched data
- Faster memory access due to CPU cache and GPU in-chip memory

## Exclusive Memory

The concept of `exclusive` memory is similar to thread-local storage, where a single variable actually has one value per thread.

In our case, we create an instance of the variable per _loop iteration_ instead of per thread.

This special _memory space_ was created from the question:

> What happens if we want data to persist across inner loops?

For example:

```okl
for (...; @outer) {
  @exclusive int id;
  for (int i = 0; i < 4; ++i; @inner) {
    id = i;
  }
  for (int i = 0; i < 4; ++i; @inner) {
    A[i] = id;
  }
}
```

would return an array `A` defined as `[0, 1, 2, 3]`

Under the hood, the code generated for regular `C++` would look like

```cpp
for (...) {
  int id[4];
  for (int i = 0; i < 4; ++i) {
    id[i] = i;
  }
  for (int i = 0; i < 4; ++i) {
    A[i] = id[i];
  }
}
```
