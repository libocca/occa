# Command-Line Interface

OCCA comes with a command-line tool called `occa` inside the `bin` directory.
It also comes with bash autocomplete to help with finding cached files.

```bash
# Autocomplete
if which occa > /dev/null 2>&1; then
    eval "$(occa autocomplete bash)"
fi
```

Calling `occa` or `occa --help` will give information about what it can do

```bash
> occa

Usage: occa [OPTIONS] COMMAND [COMMAND...]

Helpful utilities related to OCCA workflows

Commands:
  autocomplete    Prints shell functions to autocomplete occa
                  commands and arguments
  clear           Clears cached files and cache locks
  compile         Compile kernels
  env             Print environment variables used in OCCA
  info            Prints information about available backend modes
  modes           Prints available backend modes
  translate       Translate kernels
  version         Prints OCCA version

Arguments:
  COMMAND    Command to run

Options:
  -h, --help    Print usage
```

# Info

We can use `occa info` to view enabled OCCA modes and descriptions for each device.

```bash
> occa info
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

# Environment

We can use `occa env` to view the environment OCCA is using to run devices.
Environment variables override compiled-time defines.

```bash
> occa env
  Basic:
    - OCCA_DIR                   : /home/david/git/night
    - OCCA_CACHE_DIR             : [NOT SET]
    - OCCA_VERBOSE               : [NOT SET]
    - OCCA_UNSAFE                : 0
  Makefile:
    - CXX                        : g++-8
    - CXXFLAGS                   : -Wall -pedantic -Wshadow -Wsign-compare -Wuninitialized -Wtype-limits -Wignored-qualifiers -Wempty-body -Wextra -Wno-unused-parameter -Wmaybe-uninitialized -Werror -Werror=sign-compare -Werror=float-equal -g
    - FC                         : gfortran
    - FCFLAGS                    : [NOT SET]
    - LDFLAGS                    : [NOT SET]
  Backend Support:
    - OCCA_OPENMP_ENABLED        : 1
    - OCCA_CUDA_ENABLED          : 1
    - OCCA_HIP_ENABLED           : 0
    - OCCA_OPENCL_ENABLED        : 1
    - OCCA_METAL_ENABLED         : 0
  Run-Time Options:
    - OCCA_CXX                   : g++-8
    - OCCA_CXXFLAGS              : -g
    - OCCA_LDFLAGS               : [NOT SET]
    - OCCA_COMPILER_SHARED_FLAGS : [NOT SET]
    - OCCA_INCLUDE_PATH          : [NOT SET]
    - OCCA_LIBRARY_PATH          : [NOT SET]
    - OCCA_KERNEL_PATH           : [NOT SET]
    - OCCA_OPENCL_COMPILER_FLAGS : -I. -cl-single-precision-constant -cl-denorms-are-zero -cl-single-precision-constant -cl-fast-relaxed-math -cl-finite-math-only -cl-mad-enable -cl-no-signed-zeros
    - OCCA_CUDA_COMPILER         : nvcc
    - OCCA_CUDA_COMPILER_FLAGS   : -I. --compiler-options -O3 --use_fast_math
    - OCCA_HIP_COMPILER          : [NOT SET]
    - OCCA_HIP_COMPILER_FLAGS    : [NOT SET]

```

# Cache

Compiled kernels are cached in `${OCCA_CACHE_DIR}`, defaulting to `${HOME}/.occa`.
It is safe to clear the cache directory at anytime.

## Kernels

```bash
> occa clear --kernels
  Removing [/home/david/.occa/cache/*], are you sure? [y/n]:
```

## Locks

Enabling OCCA to work in distributed machines means we have to handle multiple processes across machines trying to compile the same kernel.
We use directory locks as a way to create a distributed mutex.

When processes die, OCCA catches the signal and removes locks.
However, we can remove the locks if for some reason locks still persist.

```bash
> occa clear --locks
  Removing [/home/david/.occa/locks/*], are you sure? [y/n]:
```

## All The Things!

OCCA caches other helpful files but it might be good to start with a clean environment.

```bash
> occa clear --all
  Removing [/home/david/.occa/*], are you sure? [y/n]:
```

# Translate

We can translate kernels throught the CLI which is useful for debugging

```bash
occa translate --mode 'Serial' addVectors.okl

# Short flags
occa translate -m 'Serial' addVectors.okl
```

We can inspect the launch kernel for backends that require one, such as CUDA or HIP.

```bash
occa translate --launcher --mode 'CUDA addVectors.okl

# Short flags
occa translate -lm 'CUDA' addVectors.okl
```

## Defines

We can also define macros through the `-D` flag

```bash
occa translate -D MY_VALUE=1 --mode 'Serial' addVectors.okl
```

## Include Paths

Let OCCA search for include paths with the `-I` flag

```bash
occa translate -I /path/to/kernel/header/dir --mode 'Serial' addVectors.okl
```

# Modes

Find the modes OCCA is compiled with

```bash
> occa modes
CUDA
OpenCL
OpenMP
Serial
```

# Versions

There is also a command to print versions of the OCCA API as well as the OKL parser

```bash
> occa version
1.2.0
```

```bash
> occa version --okl
1.0.12
```
