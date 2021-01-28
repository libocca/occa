# Add Vectors

**Source Code**
- [main.c](https://github.com/libocca/occa/blob/master/examples/c/1_add_vectors/main.c)
- [addVectors.okl](https://github.com/libocca/occa/blob/master/examples/c/1_add_vectors/addVectors.okl)

---

Our _'Hello World!'_ example is adding two vectors in parallel.
We'll walk through parts of the example explaining the parts and logic behind each part.

## Initializing Host Vectors

We'll start off by allocating and initializing our vectors in the host

```c
int entries = 5;

float *a  = (float*) malloc(entries * sizeof(float));
float *b  = (float*) malloc(entries * sizeof(float));
float *ab = (float*) malloc(entries * sizeof(float));

for (int i = 0; i < entries; ++i) {
  a[i]  = i;
  b[i]  = 1 - i;
  ab[i] = 0;
}
```

We will sum vectors `a` and `b` into the vector `ab`

## Initializing Device

We'll initialize our device by giving it the backend we wish to use along with information required by that backend

::: tabs backend

- Serial

    ```c
    occaDevice device = occaCreateDevice(
        occaString("mode: 'Serial'")
    );
    ```

- OpenMP

    ```c
    occaDevice device = occaCreateDevice(
        occaString("mode: 'OpenMP', threads: 4")
    );
    ```

- OpenCL

    ```c
    occaDevice device = occaCreateDevice(
        occaString("mode: 'OpenCL', device_id: 0, platform_id: 0")
    );
    ```

- CUDA

    ```c
    occaDevice device = occaCreateDevice(
        occaString("mode: 'CUDA', device_id: 0")
    );
    ```

:::

## Initializing Device Vectors

We need to allocate device memory which we'll use to add the vectors on the device

```c
occaMemory o_a  = occaDeviceMalloc(device, entries*sizeof(float), NULL, occaDefault);
occaMemory o_b  = occaDeviceMalloc(device, entries*sizeof(float), NULL, occaDefault);
occaMemory o_ab = occaDeviceMalloc(device, entries*sizeof(float), NULL, occaDefault);
```

We need to initialize `o_a` and `o_b` with the host's `a` and `b` vectors

```c
occaCopyPtrToMem(o_a, a, entries*sizeof(float), 0, occaDefault);
occaCopyPtrToMem(o_b, b, occaAllBytes         , 0, occaDefault);
```

Note we don't initialize `o_ab` since it'll be initialized when we add `o_a` and `o_b` in the device

## Build and Run Kernel

> A _kernel_ is a function on a device

We now create a kernel in the device by giving it the filename where the kernel lives, along with the kernel name.
We'll show and explain the [addVectors.okl](/examples/c/add-vectors?id=addvectorsokl) source code in the end.

```c
occaKernel addVectors = occaDeviceBuildKernel(device,
                                              "addVectors.okl",
                                              "addVectors",
                                              occaDefault);
```

We now launch the `addVectors` kernel

```c
occaKernelRun(addVectors,
              occaInt(entries), o_a, o_b, o_ab);
```

Note that we need to wrap primitive types, such as:
- `int` &nbsp; &nbsp; &nbsp; &nbsp; &rarr; &nbsp; `occaInt(value)`
- `float` &nbsp; &nbsp; &rarr; &nbsp; `occaFloat(value)`
- `double` &nbsp; &rarr; &nbsp; `occaDouble(value)`

## Copy Data to Host

To check if we added the vectors properly, we'll need to copy over the results in `o_ab` to the host

```c
occaCopyMemToPtr(ab, o_ab, occaAllBytes, 0, occaDefault);
```

## Cleaning Up

Let's make sure we got the right results

```c
for (int i = 0; i < entries; ++i) {
  if (ab[i] != (a[i] + b[i])) {
    exit(1);
  }
}
```

We need to free our host memory

```c
free(a);
free(b);
free(ab);
```

as well as our occa objects

```c
occaFree(props);
occaFree(addVectors);
occaFree(o_a);
occaFree(o_b);
occaFree(o_ab);
occaFree(device);
```

## addVectors.okl

```okl
@kernel void addVectors(const int entries,
                        const float *a,
                        const float *b,
                        float *ab) {
  for (int i = 0; i < entries; ++i; @tile(16, @outer, @inner)) {
    ab[i] = a[i] + b[i];
  }
}
```

We annotate kernels with the `@kernel` attribute and parallel loops with `@outer` and `@inner`

We use the `@tile` attribute as shorthand notation for

```okl
for (int offset = 0; offset < entries; offset += 16; @outer) {
  for (int i = offset; i < (offset + 16); ++i; @inner) {
    if (i < entries) {
      ab[i] = a[i] + b[i];
    }
  }
}
```

Note in both cases we have the conditional check `if (i < entries)`.
Since we tile by `16`, we need to make sure the iterations don't go out of bounds.
