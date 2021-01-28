# Introduction

## What is OCCA?

In a nutshell, OCCA (like *oca*-rina) is an open-source library which aims to

- Make it easy to program different types of devices (e.g. _CPU_, _GPU_, _FPGA_)
- Provide a [unified API](/guide/occa/introduction) for interacting with backend device APIs (e.g. _OpenMP_, _CUDA_, _OpenCL_)
- Use just-in-time compilation to build backend kernels
- Provide a [kernel language](/guide/okl/introduction), a minor extension to C, to abstract programming for each backend


## Getting Started

The OCCA source code can be found in [Github](https://github.com/libocca/occa) under an MIT License

Use git to download OCCA or download the latest release `.tar.gz` from the [Github releases](https://github.com/libocca/occa/releases)

```bash
git clone --depth 1 https://github.com/libocca/occa.git
cd occa
make -j4
```

We also need to add the shared library (`libocca.so`) to our linker path

::: tabs os

- Linux

    ```bash
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${PWD}/lib"
    ```

- MacOS

    ```bash
    export DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH}:${PWD}/lib"
    ```
:::

## Testing Installation

OCCA comes with a command called `occa`, found inside the `bin` directory.
The purpose of `occa` is to help gather device information as well as other useful utilities

For more information, please see the [Command-Line Interface](/guide/user-guide/command-line-interface) section.

```bash
./bin/occa info
```

Output:

```bash
    ========+======================+=========================================
     CPU(s) | Processor Name       | Intel(R) Core(TM) i7-7700 CPU @ 3.60GHz
            | Memory               | 62.84 GB
            | Clock Frequency      | 4.2 MHz
            | SIMD Instruction Set | SSE2
            | SIMD Width           | 128 bits
            | L1d Cache Size       |  32 KB
            | L1i Cache Size       |  32 KB
            | L2 Cache Size        | 256 KB
            | L3 Cache Size        |   8 MB
    ========+======================+=========================================
     OpenCL | Device Name          | GeForce GTX 1080
            | Driver Vendor        | NVIDIA
            | Platform ID          | 0
            | Device ID            | 0
            | Memory               | 7.92 GB
    ========+======================+=========================================
     CUDA   | Device Name          | GeForce GTX 1080
            | Device ID            | 0
            | Memory               | 7.92 GB
    ========+======================+=========================================
```

## Running an Example

The simplest example is to create and add two vectors together

```bash
cd examples/cpp/01_add_vectors
make
./main --verbose
```

```bash
Compiling [addVectors]
g++-8 -g -std=c++11 -fPIC -shared /home/david/.occa/cache/3cf53e2af2c17b3e/source.cpp -o /home/david/.occa/cache/3cf53e2af2c17b3e/binary -I/home/david/git/night/include -I/home/david/git/night/include -L/home/david/git/night/lib -locca
0: 1
1: 1
2: 1
3: 1
4: 1
```

## Kernel Caching

Note that when running `addVectors` is run a second time, the compilation info changes.
Compiled kernels are cached and its binaries are reused if nothing changed in the compilation step (e.g. device information, kernel defines, etc)

?> Compiled kernels are cached for fast consecutive builds!

First run

```bash
Compiling [addVectors]
g++-8 -g -std=c++11 -fPIC -shared /home/david/.occa/cache/3cf53e2af2c17b3e/source.cpp -o /home/david/.occa/cache/3cf53e2af2c17b3e/binary -I/home/david/git/night/include -I/home/david/git/night/include -L/home/david/git/night/lib -locca
```

Second run

```bash
Loading cached [addVectors] from [/home/david/git/night/examples/cpp/01_add_vectors/addVectors.okl] in [3cf53e2af2c17b3e/binary]
```
