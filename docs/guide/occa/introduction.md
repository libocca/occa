# Introduction

The design of the OCCA (like *oca*-rina) API is based on the functionality of physical devices.
The main two being

- Memory management
- Code execution

We generalize different device architectures by wrapping these two concepts in a single API.
We'll showcase the basics by going through a simple example where we add 2 vectors.

### Terminology

<span style="font-size: 1.1em">_Host_</span>
<template><div style="margin-top: -0.8em; padding-left: 1em;">
The physical device running the application code.
This is usually a CPU processor.
</div></template>

<span style="font-size: 1.1em">_Device_</span>
<template><div style="margin-top: -0.8em; padding-left: 1em;">
A physical device we're communicating with, whether the same physical device as the host or an offload device.
Examples include CPU processors, GPUs, and Xeon Phi.
</div></template>


<span style="font-size: 1.1em">_Kernel_</span>
<template><div style="margin-top: -0.8em; padding-left: 1em;">
A function that runs on a device
</div></template>

# Device

We start off by connecting to a physical device through an OCCA device object.

::: tabs language

- C++

    ```cpp
    #include "occa.hpp"

    occa::device device("mode: 'Serial'");
    ```

- C

    ```c
    #include "occa.h"

    occaDevice device = occaCreateDevice("mode: 'Serial'");
    ```

:::

The string used to initialize the device

```cpp
"mode: 'Serial'"
```

creates an `occa::properties` object which is then passed to the device.
Properties uses JSON format with some shorthand notations found in JavaScript.

```js
{
  mode: 'Serial'
}
```

The only property field required by OCCA when creating a device is `mode`.
However, each `mode` has its own requirements, such as CUDA requiring a `deviceID`.
In this case, we're initializing a device that runs code serially.

?> **Serial** mode is useful for debugging kernel code by enabling the use of debuggers such as _lldb_ or _gdb_

Here are examples for the all core modes supported in OCCA.

::: tabs backend

- Serial

    ```cpp
    "mode: 'Serial'"
    ```

    ```js
    {
      mode: 'Serial'
    }
    ```

- OpenMP

    ```cpp
    "mode: 'OpenMP', threads: 4"
    ```

    ```js
    {
      mode: 'OpenMP',
      threads: 4
    }
    ```

- Threads

    ```cpp
    "mode: 'Serial', threads: 4, pinnedCores: [0, 1, 2, 3]"
    ```

    ```js
    {
      mode: 'Serial',
      threads: 4,
      pinnedCores: [0, 1, 2, 3]
    }
    ```

- OpenCL

    ```cpp
    "mode: 'OpenCL', deviceID: 0, platformID: 0"
    ```

    ```js
    {
      mode: 'OpenCL',
      deviceID: 0,
      platformID: 0
    }
    ```

- CUDA

    ```cpp
    "mode: 'CUDA', deviceID: 0"
    ```

    ```js
    {
      mode: 'CUDA',
      deviceID: 0
    }
    ```

:::

# Memory

Now that we have a device, we need to allocate and initialize memory in the device.
We cannot modify device memory directly in the host, hence data is usually initialized either by:

- Copying host data to the device
- Modifying device data inside a kernel running on the device

For our case, we'll initialize data in the host and copy it to the device.

::: tabs language

- C++

    ```cpp
    const int entries = 5;
    float *a  = new float[entries];
    float *b  = new float[entries];
    float *ab = new float[entries]

    for (int i = 0; i < entries; ++i) {
      a[i]  = i;
      b[i]  = 1 - i;
      ab[i] = 0;
    }
    ```

- C

    ```c
    const int entries = 5;
    float *a  = (float*) malloc(entries * sizeof(float));
    float *b  = (float*) malloc(entries * sizeof(float));
    float *ab = (float*) malloc(entries * sizeof(float));

    for (int i = 0; i < entries; ++i) {
      a[i]  = i;
      b[i]  = 1 - i;
      ab[i] = 0;
    }
    ```

:::

We now allocate and initialize device memory using our host arrays.

We initialize both `o_a` and `o_b` by copying over data from `a` and `b` respectively.

Since we'll be computing `o_ab` by summing `o_a` and `o_b`, there is no need to transfer `ab` to the device.

::: tabs language

- C++

    ```cpp
    // Copy a and b to the device
    occa::memory o_a  = device.malloc(entries * sizeof(float), a);
    occa::memory o_b  = device.malloc(entries * sizeof(float), b);
    // Don't initialize o_ab
    occa::memory o_ab = device.malloc(entries * sizeof(float));
    ```

- C

    ```c
    // Copy a and b to the device
    occaMemory o_a  = occaDeviceMalloc(device,
                                       entries * sizeof(float),
                                       a, occaDefault);
    occaMemory o_b  = occaDeviceMalloc(device,
                                       entries * sizeof(float),
                                       b, occaDefault);
    // Don't initialize o_ab
    occaMemory o_ab = occaDeviceMalloc(device,
                                       entries * sizeof(float),
                                       NULL, occaDefault);
    ```

:::

# Kernel

Now that we have data in the device, we want to start computing our vector addition.
We must first build a kernel that uses device data.

Kernels are built at runtime so we require 2 things:

- The file with the kernel source code
- The name of the kernel in the source code we wish to use

::: tabs language

- C++

    ```cpp
    occa::kernel addVectors = device.buildKernel("addVectors.okl",
                                                 "addVectors");
    ```

- C

    ```c
    occaKernel addVectors = occaDeviceBuildKernel(device,
                                                  "addVectors.okl",
                                                  "addVectors");
    ```

:::


We can now call `addVectors` with our device arrays.

::: tabs language

- C++

    ```cpp
    addVectors(entries, o_a, o_b, o_ab);
    ```

- C

    ```c
    occaKernelRun(addVectors,
                  occaInt(entries),
                  o_a, o_b, o_ab);
    ```

:::

# Syncing Host and Device

Kernel launches are not guaranteed to be blocking in all modes.
For example, OpenCL and CUDA both launch their kernels in a non-blocking fashion.

We'll show a few ways to sync the device with the host, the first being the `finish` method:

::: tabs language

- C++

    ```cpp
    device.finish();
    ```

- C

    ```c
    occaDeviceFinish(device);
    ```

:::


The second way we can sync the device and host is through blocking copies to/from the device:

::: tabs language

- C++

    ```cpp
    occa::memcpy(b, o_b);
    ```

- C

    ```c
    occaCopyMemToPtr(b, o_b,
                     occaAllBytes, 0,
                     occaDefault);
    ```

:::


In this case, we'll use the `memcpy` approach to sync the host and look at `o_ab` through the host's `ab` pointer.

::: tabs language

- C++

    ```cpp
    for (int i = 0; i < entries; ++i) {
      if (ab[i] != 1) {
        std::cerr << "addVectors failed!!\n";
      }
    }
    ```

- C

    ```c
    for (int i = 0; i < entries; ++i) {
      if (ab[i] != 1) {
        fprintf(stderr, "addVectors failed!!\n");
      }
    }
    ```

:::

# Garbage Collection

?> In C++, each `occa` object uses reference counting to automatically free itself.
Manually freeing an object won't cause a double free.

Freeing must be done manually in C since there is no concept of an automatic destructor.
However, we tried to make it as easy as possible to free `occa` objects.

```c
occaFree(device);
occaFree(kernel);
occaFree(memory);
occaFree(properties);
occaFree(stream);
```

For more information, checkout the API sections.
