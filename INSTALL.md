# INSTALLATION GUIDE 

## Requirements

### Minimum

- [CMake] v3.17 or newer
- C++17 compiler
- C11 compiler

### Optional

 - Fortan 90 compiler
 - CUDA 9 or later
 - HIP 3.5 or later
 - SYCL 2020 or later
 - OpenCL 2.0 or later
 - OpenMP XXX

## Configure

OCCA uses the [CMake] build system. For convenience, the shell script `configure-cmake.sh` has been provided to drive the Cmake build. The following table gives a list of build parameters which are set in the file. To override the default value, it is only necessary to assign the variable an alternate value at the top of the script or at the commandline.

Example
```shell
$> CC=clang CXX=clang++ ENABLE_OPENMP="OFF" ./configure-cmake.sh
```

### Build Parameters  

| Parameter | Description | Default |
| --------- | ----------- | ------- |
| BUILD_DIR | Directory used by CMake to build OCCA | `./build` |
| INSTALL_DIR | Directory where OCCA should be installed | `./install` |
| BUILD_TYPE | `Debug`, `RelWithDebInfo`, `Release` | `RelWithDebInfo` |
| PREFIX_PATHS | Semicolon separated list of paths to 3rd-party dependencies | *empty*
| CXX | C++11 compiler | `g++` |
| CXXFLAGS | C++ compiler flags | *empty* | 
| CC | C11 compiler| `gcc` |
| CFLAGS | C compiler flags | *empty* |
| ENABLE_CUDA | Enable use of the CUDA backend | `ON`|
| ENABLE_HIP | Enable use of the HIP backend | `ON`|
| ENABLE_DPCPP | Enable use of the DPC++ backend | `ON`|
| ENABLE_OPENCL | Enable use of the OpenCL backend | `ON`|
| ENABLE_OPENMP | Enable use of the OpenMP backend | `ON`|
| ENABLE_METAL | Enable use of the Metal backend | `ON`|
| ENABLE_TESTS | Build OCCA's test harness | `ON` |
| ENABLE_EXAMPLES | Build OCCA examples | `ON` |
| ENABLE_FORTRAN | Build the Fortran language bindings | `OFF`|
| FC | Fortran 90 compiler | `gfortran` |
| FFLAGS | Fortran compiler flags | *empty* |

## Building  

...

## Testing  

CTest is used for the OCCA test harness and can be run using the command
```shell
$> ctest --test-dir BUILD_DIR --output-on-failure
```

Before running CTest, it is important to set the environment variables `OCCA_CXX` and `OCCA_CC`; otherwise, tests for some backends may return a false negative.

For testing, `BUILD_DIR/occa` is used for kernel caching. It may be necessary to clear this directory when rerunning tests after rebuilding with an existing configuration.

## Installation  


[CMake]: https://cmake.org/
