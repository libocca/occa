# Getting Started

The OCCA source code can be found in |Github Link| under an MIT License

Below are the list of dependencies

::: tabs os

- Linux

    - C++ compiler
    - `make`

- MacOS

    - C++ compiler
    - `make`

:::


# Download and Installation

Use git to download OCCA or download the latest release `.tar.gz` from |Github Release Link|

::: tabs os

- Linux

    ```bash
    git clone https://github.com/libocca/occa.git
    cd occa
    make -j4
    ```

- MacOS

    ```bash
    git clone https://github.com/libocca/occa.git
    cd occa
    make -j4
    ```

:::


# Testing Installation

OCCA comes with a command called `occa`, found inside the `bin` directory.
The purpose of `occa` is to help gather device information as well as other useful utilities

For more information, please see the [Command-Line Interface](getting-started) section. <!-- TODO -->

::: tabs os

- Linux

    ```bash
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${PWD}/lib"
    PATH+=":${PWD}/bin"
    occa info
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

- MacOS

    ```bash
    export DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH}:${PWD}/lib"
    PATH+=":${PWD}/bin"
    occa info
    ```

    Output

    ```bash
    ==========+======================+==========================================
     CPU Info | Processor Name       | Intel(R) Core(TM) i5-4260U CPU @ 1.40GHz
              | Cores                | 4
              | Memory (RAM)         | 4 GB
              | Clock Frequency      | 1.4 GHz
              | SIMD Instruction Set | SSE4
              | SIMD Width           | 128 bits
              | L1 Cache Size (d)    |  32 KB
              | L2 Cache Size        | 256 KB
              | L3 Cache Size        |   3 MB
    ==========+======================+==========================================
     OpenCL   | Device Name          | Intel(R) Core(TM) i5-4260U CPU @ 1.40GHz
              | Driver Vendor        | Intel
              | Platform ID          | 0
              | Device ID            | 0
              | Memory               | 4 GB
              |----------------------+------------------------------------------
              | Device Name          | HD Graphics 5000
              | Driver Vendor        | Intel
              | Platform ID          | 0
              | Device ID            | 1
              | Memory               | 1 GB
    ==========+======================+==========================================
    ```

:::

# Example

The simplest example is to create and add two vectors together

::: tabs os

- Linux

    ```bash
    cd examples/1_add_vectors/cpp
    make
    OCCA_VERBOSE=1 ./main
    ```

    Output

    ```bash
    ==========+======================+==========================================
     CPU Info | Cores                | 16
              | Memory (RAM)         | 31 GB
              | Clock Frequency      | 32.684 GHz
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
    Compiling [addVectors]
    clang++ -x c++ -fPIC -shared -g /home/dsm5/.occa/cache/e60679bfca62c2f2/source.occa -o /home/dsm5/.occa/cache/e60679bfca62c2f2/binary -I/home/dsm5/git/night/include -L/home/dsm5/git/night/lib -locca
    0: 1
    1: 1
    2: 1
    3: 1
    4: 1
    ```

- MacOS

    ```bash
    cd examples/1_add_vectors/cpp
    make
    OCCA_VERBOSE=1 ./main
    ```

    Output

    ```bash
    ==========+======================+==========================================
     CPU Info | Processor Name       | Intel(R) Core(TM) i5-4260U CPU @ 1.40GHz
              | Cores                | 4
              | Memory (RAM)         | 4 GB
              | Clock Frequency      | 1.4 GHz
              | SIMD Instruction Set | SSE4
              | SIMD Width           | 128 bits
              | L1 Cache Size (d)    |  32 KB
              | L2 Cache Size        | 256 KB
              | L3 Cache Size        |   3 MB
    ==========+======================+==========================================
     OpenCL   | Device Name          | Intel(R) Core(TM) i5-4260U CPU @ 1.40GHz
              | Driver Vendor        | Intel
              | Platform ID          | 0
              | Device ID            | 0
              | Memory               | 4 GB
              |----------------------+------------------------------------------
              | Device Name          | HD Graphics 5000
              | Driver Vendor        | Intel
              | Platform ID          | 0
              | Device ID            | 1
              | Memory               | 1 GB
    ==========+======================+==========================================
    Compiling [addVectors]
    clang++ -x c++ -fPIC -shared -I. -D__extern_always_inline=inline -O3 -mtune=native -ftree-vectorize -funroll-loops -ffast-math /Users/dsm5/.occa/cache/4c38ebbf648a4b23/source.occa -o /Users/dsm5/.occa/cache/4c38ebbf648a4b23/binary -I/Users/dsm5/git/night/include -L/Users/dsm5/git/night/lib -locca
    0: 1
    1: 1
    2: 1
    3: 1
    4: 1
    ```

:::


# Kernel Caching

?> Compiled kernels are cached for fast consecutive builds!

Note that when running `addVectors` is run a second time, the compilation info changes.
Compiled kernels are cached and its binaries are reused if nothing changed in the compilation step (e.g. device information, kernel defines, etc)


::: tabs os

- Linux

    First run

    ```bash
    Compiling [addVectors]
    clang++ -x c++ -fPIC -shared -g /home/dsm5/.occa/cache/e60679bfca62c2f2/source.occa -o /home/dsm5/.occa/cache/e60679bfca62c2f2/binary -I/home/dsm5/git/night/include -L/home/dsm5/git/night/lib -locca
    ```

    Second run

    ```bash
    Loading cached [addVectors0] from [2d38aae833d7e36a/parsed-source.cpp] in [e60679bfca62c2f2/device-binary]
    ```

- MacOS

    First run

    ```bash
    Compiling [addVectors]
    clang++ -x c++ -fPIC -shared -I. -D__extern_always_inline=inline -O3 -mtune=native -ftree-vectorize -funroll-loops -ffast-math /Users/dsm5/.occa/cache/4c38ebbf648a4b23/source.occa -o /Users/dsm5/.occa/cache/4c38ebbf648a4b23/binary -I/Users/dsm5/git/night/include -L/Users/dsm5/git/night/lib -locca
    ```

    Second run

    ```bash
    Loading cached [addVectors0] from [2d38aae833d7e36a/parsed-source.cpp] in [e60679bfca62c2f2/device-binary]
    ```

:::


<!-- .. |Github Link| raw:: html -->

<!--    <a href="https://github.com/libocca/occa" target="_blank"> -->
<!--       <i class="github icon" style="margin-right: 0;"></i> -->
<!--       Github -->
<!--    </a> -->

<!-- .. |Github Release Link| raw:: html -->

<!--    <a href="https://github.com/libocca/occa/releases" target="_blank"> -->
<!--       <i class="github icon" style="margin-right: 0;"></i> -->
<!--       Github -->
<!--    </a> -->
