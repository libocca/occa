<a name="OCCA"></a>
## OCCA

[![Build Status](https://travis-ci.org/libocca/occa.svg?branch=master)](https://travis-ci.org/libocca/occa)
[![Join the chat at https://gitter.im/libocca/occa](https://badges.gitter.im/libocca/occa.svg)](https://gitter.im/libocca/occa?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

OCCA is an open-source (MIT license) library used to program current multi-core/many-core architectures.
Devices (such as CPUs, GPUs, Intel's Xeon Phi, FPGAs, etc) are abstracted using an offload-model for application development and programming for the devices is done through a C-based (OKL) or Fortran-based kernel language (OFL).
OCCA gives developers the ability to target devices at run-time by using run-time compilation for device kernels.

<a name="Taiga"></a>
## Taiga

We are using <a href="https://tree.taiga.io/project/dsm5-occa/kanban">Taiga</a> to schedule and manage our tasks.

To submit an issue, please follow <a href="https://tree.taiga.io/project/dsm5-occa/issues?page=1">this link</a>

<a name="README"></a>
## README

### Installing

Using a terminal, go to your OCCA directory
You should see:
   README.md (this file)
   include
   src
   lib

To compile `libocca.so`, type:

```
make
```

To compile the Fortran library, setup the `OCCA_FORTRAN_ENABLED` environment variable before compiling

```
export OCCA_FORTRAN_ENABLED="1"
```

Python 2 and 3 bindings are available with OCCA.
If you wish to setup the occa Python module, rather than using `make`, compile both `libocca.so` and the module with

```
python make.py
```


### Examples

We have a few examples to show different features of OCCA. The addVectors example contains examples for each current supported language

* C++
* C
* Python
* Fortran
* Julia

#### Compile
To compile addVectors (Hello World! style example) in C++

```
cd examples/addVectors/cpp
make
```

#### Environment
Setup your `LD_LIBRARY_PATH` to point to libocca.so

```
export LD_LIBRARY_PATH+=':<occa>/lib'
```
where `<occa>` is the OCCA directory

### Run
```
./main
```

### Status
* Linux and OSX are fully supported
* Windows is partially supported
  * Code is up-to-date for windows
  * Missing compilation project/scripts
  * Visual Studio project is out of date

* OKL Status:
  * Supports most of C (send bugs =))
  * Preprocessor is missing variadic functions

* OFL Status:
  * Obsolete for now
  * Version 0.1 supports a subset of Fortran:
    * integer, real, character, logical, double precision
    * function, subroutine
    * DO, WHILE, IF, IF ELSE, ELSE

### Useful environment variables:
| Environment Variable       | Description                                         |
|----------------------------|-----------------------------------------------------|
| OCCA_CACHE_DIR             | Sets directory where kernels are cached (Default: ${HOME}/._occa |
| OCCA_INCLUDE_PATH          | Adds directories to find headers |
| OCCA_LIBRARY_PATH          | Adds directories to find libraries |
| OCCA_CXX                   | C++ compiler used for libocca.so and run-time compilation |
| OCCA_CXXFLAGS              | C++ compiler flags used for libocca.so and run-time compilation |

#### OpenCL
| Environment Variable       | Description                                         |
|----------------------------|-----------------------------------------------------|
| OCCA_OPENCL_COMPILER_FLAGS | Adds additional OpenCL flags when compiling kernels |

#### CUDA
| Environment Variable       | Description                                         |
|----------------------------|-----------------------------------------------------|
| OCCA_CUDA_COMPILER         | Can be used to specify the CUDA compiler            |
| OCCA_CUDA_COMPILER_FLAGS   | Adds additional OpenCL flags when compiling kernels |