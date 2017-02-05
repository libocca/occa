<a name="OCCA"></a>
# OCCA

[![Build Status](https://travis-ci.org/libocca/occa.svg?branch=master)](https://travis-ci.org/libocca/occa)
[![Documentation](https://readthedocs.org/projects/occa/badge/?version=latest)](https://occa.readthedocs.io/en/latest/?badge=latest)
[![Join the chat at https://gitter.im/libocca/occa](https://badges.gitter.im/libocca/occa.svg)](https://gitter.im/libocca/occa?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

OCCA is an open-source (MIT license) library used to program current multi-core/many-core architectures.
Devices (such as CPUs, GPUs, Intel's Xeon Phi, FPGAs, etc) are abstracted using an offload-model for application development and programming for the devices is done through a C-based (OKL) or Fortran-based kernel language (OFL).
OCCA gives developers the ability to target devices at run-time by using run-time compilation for device kernels.

## Installing

Using a terminal, go to your OCCA directory
You should see:
   README.md (this file)
   include
   src
   lib

To compile `libocca.so`, type:

```bash
git clone https://github.com/libocca/occa.git
cd occa
make -j 4
```

To compile the Fortran library, setup the `OCCA_FORTRAN_ENABLED` environment variable before compiling

```bash
export OCCA_FORTRAN_ENABLED="1"
```

Python 2 and 3 bindings are also available.
If you wish to setup the occa Python module

```bash
python scripts/make.py
```


## Examples

We have a few examples to show different features of OCCA. The addVectors example contains examples for each current supported language

* C++
* C
* Python
* Fortran
* Julia

### Compile
To compile addVectors (Hello World! style example) in C++

```bash
cd examples/addVectors/cpp
make
```

### Environment
Setup your library path to point to libocca.so

```bash
# Linux
export LD_LIBRARY_PATH+=':<occa>/lib'
# Mac
export DYLD_LIBRARY_PATH+=':<occa>/lib'
```
where `<occa>` is the OCCA directory

## Run
```bash
./main
```

## Status
* Linux and OSX are fully supported
* Windows is partially supported
  * Code is up-to-date for windows
  * Missing compilation project/scripts
  * Visual Studio project is out of date

* OKL Status:
  * Supports most of C

## Useful environment variables:
| Environment Variable       | Description                                         |
|----------------------------|-----------------------------------------------------|
| OCCA_DIR                   | Sets directory where OCCA was installed |
| OCCA_CACHE_DIR             | Sets directory where kernels are cached (Default: ${HOME}/._occa |
| OCCA_INCLUDE_PATH          | Adds directories to find headers |
| OCCA_LIBRARY_PATH          | Adds directories to find libraries |
| OCCA_CXX                   | C++ compiler used for libocca.so and run-time compilation |
| OCCA_CXXFLAGS              | C++ compiler flags used for libocca.so and run-time compilation |

### OpenCL
| Environment Variable       | Description                                         |
|----------------------------|-----------------------------------------------------|
| OCCA_OPENCL_COMPILER_FLAGS | Adds additional OpenCL flags when compiling kernels |

### CUDA
| Environment Variable       | Description                                         |
|----------------------------|-----------------------------------------------------|
| OCCA_CUDA_COMPILER         | Can be used to specify the CUDA compiler            |
| OCCA_CUDA_COMPILER_FLAGS   | Adds additional OpenCL flags when compiling kernels |