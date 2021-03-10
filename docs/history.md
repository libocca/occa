# History

OCCA (like *oca*-rina) started off as a project in [Tim Warburton's group](https://www.paranumal.com).
The group mainly worked on high-order numerical methods, specifically on the high performance algorithms.
During that time, we mainly focused on GPGPU programming using OpenCL and CUDA.

We had wrappers for OpenCL and CUDA to test implementations, which we almost always had 2 almost identical codes to run on NVIDIA and AMD GPUs.
In the summer of 2014, we tried out merging the wrappers (+ a new OpenMP one) and use just-in-time (JIT) compilation to generate the kernels.
We still faced the problem that we had duplicate code for the kernels.

### OCCA v1

The first approach to solving this used compiler macros that were defined based on the backend.
If curious, this approach is still documented in the [original OCCA paper](https://arxiv.org/abs/1403.0968)
Internally in the group, we call this **OCCA v1**.

The _Hello World_ of kernels in **OCCA v1**

```cpp
occaKernel void addVectors(occaKernelInfoArg,
                           const int occaVariable entries,
                           occaPointer const float *a,
                           occaPointer const float *b,
                           occaPointer float *ab) {
    occaOuterFor0{
      occaInnerFor0{
        int i = occaInnerId0 + (occaInnerDim0 * occaOuterId0)
        if(i < entries) {
          ab[i] = a[i] + b[i];
        }
      }
    }
  }
}
```

with the application code

```cpp
  // Setup device
  occa dev;
  dev.setup("CPU", platform, device);

  // Build kernel
  occaKernel addVectors = dev.buildKernel("simple.occa",
                                          "simple",
                                          " ");

  // Set kernel dimensions
  int dim = 1;
  size_t outer[3] = {100 * 128, 1, 1};
  size_t inner[3] = {128, 1, 1};
  addVectors.setThreadArray(outer, inner, dim);

  // Allocate memory
  occaMemory o_a = dev.createBuffer(N*sizeof(double), NULL);

  // Launch kernel
  addVectors(entries, o_a, o_b, o_ab);
```

?> Although the idea has stayed the same throughout all these years, we've been refining the library during that time

At this point, the project was only used within the [research group](https://www.paranumal.com).
As we started collaborating with more folks, we wanted to make the kernel code more user friendly.

Cue **OCCA v2**

### OCCA v2

The API found in **OCCA v2** was also complete refactor from **OCCA v1**, much closer to the current OCCA API.
The code is still available in the [v0.2.0 tag](https://github.com/libocca/occa/releases/tag/v0.2.0).

How we program the OCCA kernels became the biggest distinction from **OCCA v1**.
Rather than using the macro approach, we pivoted to using a parser and apply code transformations.

We call this small extension of C the _OCCA Kernel Language_ (OKL).

```okl
kernel void addVectors(const int entries,
                       const float *a,
                       const float *b,
                       float *ab) {

  for (int i = 0; i < entries; ++i; tile(16)) {
    if (i < entries) {
      ab[i] = a[i] + b[i];
    }
  }
}
```

If you're already familiar with the current version of OKL, you might notice

- `kernel` &nbsp; &nbsp; &nbsp; &rarr; &nbsp; `@kernel`
- `tile(16)` &nbsp; &rarr; &nbsp; `@tile(16, @outer, @inner)`

The OKL parser was far from perfect and occasional syntax errors would cause segfaults.
But it was a fun project that made kernels easier to read and could bring people in the research group up to speed faster.

In 2017, we joined the [Center for Efficient Exascale Discretizations (CEED)](http://ceed.exascaleproject.org/) team.

### OCCA

CEED is one of the few co-design centers within the U.S. Department of Energy (DOE) Exascale Computing Project (ECP).
OCCA is _one of the solutions_ being looked at for enabling exascale computing in high-order numerical methods.

?> We needed to step up from a _research group_ project to a _production-grade_ project

The API was revisited and updated once again ([v0.2 -> v1.0 Guide](https://github.com/libocca/occa/releases/tag/v1.0.0-alpha.1#porting-from-v0.2-to-v1.0)), with the focus on

- &#8317;&sup1;&#8318; No dependencies
- Minimalist and simple API
- Enable backend-specific features
- Able to extend backends outside of the OCCA repo
- Able to extend the kernel language outside of the OCCA repo

> &#8317;&sup1;&#8318; This is to make it easy for users to install and the lack of a good C++ package manager

### The Present

> If you're looking to learn or contribute, feel free to reach out and I'll be more than happy to fill you in to any questions you might have :)
>
> \- [David Medina](https://github.com/dmed256)
