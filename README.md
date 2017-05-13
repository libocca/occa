<a name="OCCA"></a>
# OCCA

[![Build Status](https://travis-ci.org/libocca/occa.svg?branch=master)](https://travis-ci.org/libocca/occa)
[![Coverage Status](https://coveralls.io/repos/github/libocca/occa/badge.svg?branch=1.0)](https://coveralls.io/github/libocca/occa?branch=1.0)
[![Documentation](https://readthedocs.org/projects/occa/badge/?version=latest)](https://occa.readthedocs.io/en/latest/?badge=latest)

OCCA is an open-source (MIT license) library used to program current multi-core/many-core architectures.
Devices (such as CPUs, GPUs, Intel's Xeon Phi, FPGAs, etc) are abstracted using an offload-model for application development and programming for the devices is done through a C-based (OKL) kernel.
OCCA gives developers the ability to target devices at run-time by using run-time compilation for device kernels.

### Installing

```bash
git clone https://github.com/libocca/occa.git
cd occa
make -j 4
```

### Environment

Setup environment variables inside the `occa` directory

#### Linux

```bash
export OCCA_DIR="${PWD}"
export PATH+=":${OCCA_DIR}"
export LD_LIBRARY_PATH+=":${OCCA_DIR}/lib"
```

#### Mac OSX

```bash
export OCCA_DIR="${PWD}"
export PATH+=":${OCCA_DIR}"
export DYLD_LIBRARY_PATH+=":${OCCA_DIR}/lib"
```

### Hello World

```bash
cd ${OCCA_DIR}/examples/addVectors/cpp
make
./main
```

### occa Command

There is an executable `occa` provided inside `${OCCA_DIR}/bin`

```bash
> occa --help

Usage: occa COMMAND

Can be used to display information of cache kernels.

Commands:
  autocomplete    Prints shell functions to autocomplete occa
                  commands and arguments
  cache           Cache kernels
  clear           Clears cached files and cache locks
  env             Print environment variables used in OCCA
  info            Prints information about available OCCA modes

Arguments:
  COMMAND    Command to run
```

#### Linux Autocomplete

```bash
. <(occa autocomplete bash)
```

#### Mac OSX Autocomplete

```bash
eval "$(occa autocomplete bash)"
```

### Useful environment variables:
| Environment Variable       | Description                                                                         |
|----------------------------|-------------------------------------------------------------------------------------|
| OCCA_DIR                   | Directory where OCCA is installed, overwrites `occa::OCCA_DIR` set at compile-time  |
| OCCA_CACHE_DIR             | Directory where kernels get cached (Default: `${HOME}/.occa`)                       |
| OCCA_INCLUDE_PATH          | Path to find headers, such as CUDA and OpenCL headers (`:` delimited)               |
| OCCA_LIBRARY_PATH          | Path to find .so libraries (`:` delimited)                                          |
| OCCA_CXX                   | C++ compiler used for run-time kernel compilation                                   |
| OCCA_CXXFLAGS              | C++ compiler flags used for run-time kernel compilation                             |
| OCCA_OPENCL_COMPILER_FLAGS | Additional OpenCL flags when compiling kernels                                      |
| OCCA_CUDA_COMPILER         | Compiler used for run-time CUDA kernel compilation                                  |
| OCCA_CUDA_COMPILER_FLAGS   | CUDA compiler flags used for run-time kernel compilation                            |
| OCCA_VERBOSE               | Verbose logging is suppresed if set to either: `0, n, no, false` (Default: `false`) |
