# Getting Started

The OCCA source code can be found in [Github](https://github.com/libocca/occa) under an MIT License

Below are the list of dependencies

- C++ compiler
- `make`


# Download and Installation

Use git to download OCCA or download the latest release `.tar.gz` from the [Github releases](https://github.com/libocca/occa/releases)

```bash
git clone https://github.com/libocca/occa.git
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

# Testing Installation

OCCA comes with a command called `occa`, found inside the `bin` directory.
The purpose of `occa` is to help gather device information as well as other useful utilities

For more information, please see the [Command-Line Interface](user-guide/command-line-interface) section.

```bash
./bin/occa info
```

Output:

```bash
==========+======================+==========================================
 CPU Info | Cores                | 16
          | Memory (RAM)         | 31 GB
          | Clock Frequency      | 32.623 GHz
          | SIMD Instruction Set | SSE2
          | SIMD Width           | 128 bits
==========+======================+==========================================
 OpenCL   | Device Name          | GeForce GTX 980
          | Driver Vendor        | NVIDIA
          | Platform ID          | 0
          | Device ID            | 0
          | Memory               | 3 GB
          |----------------------+------------------------------------------
          | Device Name          | GeForce GTX 980
          | Driver Vendor        | NVIDIA
          | Platform ID          | 0
          | Device ID            | 1
          | Memory               | 3 GB
          |----------------------+------------------------------------------
          | Device Name          | GeForce GTX 980
          | Driver Vendor        | NVIDIA
          | Platform ID          | 0
          | Device ID            | 2
          | Memory               | 3 GB
          |----------------------+------------------------------------------
          | Device Name          | Intel(R) Core(TM) i7-5960X CPU @ 3.00GHz
          | Driver Vendor        | Intel
          | Platform ID          | 1
          | Device ID            | 0
          | Memory               | 31 GB
==========+======================+==========================================
 CUDA     | Device ID            | 0
          | Device Name          | GeForce GTX 980
          | Memory               | 3 GB
          |----------------------+------------------------------------------
          | Device ID            | 1
          | Device Name          | GeForce GTX 980
          | Memory               | 3 GB
          |----------------------+------------------------------------------
          | Device ID            | 2
          | Device Name          | GeForce GTX 980
          | Memory               | 3 GB
==========+======================+==========================================
```

# Example

The simplest example is to create and add two vectors together

```bash
cd examples/1_add_vectors/cpp
make
OCCA_VERBOSE=1 ./main
```

```bash
Compiling [addVectors]
clang++ -x c++ -fPIC -shared -I. -D__extern_always_inline=inline -O3 -mtune=native -ftree-vectorize -funroll-loops -ffast-math /home/david/.occa/cache/4c38ebbf648a4b23/source.occa -o /home/david/.occa/cache/4c38ebbf648a4b23/binary -I/home/david/git/night/include -L/home/david/git/night/lib -locca
0: 1
1: 1
2: 1
3: 1
4: 1
```

# Kernel Caching

Note that when running `addVectors` is run a second time, the compilation info changes.
Compiled kernels are cached and its binaries are reused if nothing changed in the compilation step (e.g. device information, kernel defines, etc)

?> Compiled kernels are cached for fast consecutive builds!

First run

```bash
Compiling [addVectors]
clang++ -x c++ -fPIC -shared -I. -D__extern_always_inline=inline -O3 -mtune=native -ftree-vectorize -funroll-loops -ffast-math /home/david/.occa/cache/4c38ebbf648a4b23/source.occa -o /home/david/.occa/cache/4c38ebbf648a4b23/binary -I/home/david/git/night/include -L/home/david/git/night/lib -locca
```

Second run

```bash
Loading cached [addVectors0] from [2d38aae833d7e36a/parsed-source.cpp] in [e60679bfca62c2f2/device-binary]
```
