# INSTALLATION GUIDE 

## Requirements

### Minimum

- [CMake] v3.21 or newer
- C++17 compiler
- C11 compiler

### Optional

 - Fortan 90 compiler
 - CUDA 9 or later
 - HIP 3.5 or later
 - SYCL 2020 or later
 - OpenCL 2.0 or later
 - OpenMP 4.0 or later
 - Support Clang based transpiler

## Linux

### **Configure**

OCCA uses the [CMake] build system. For convenience, the shell script `configure-cmake.sh` has been provided to drive the Cmake build. The following table gives a list of build parameters which are set in the file. To override the default value, it is only necessary to assign the variable an alternate value at the top of the script or at the commandline.

Example
```shell
$ CC=clang CXX=clang++ OCCA_ENABLE_OPENMP="OFF" ./configure-cmake.sh
``` 

| Build Parameter | Description | Default |
| --------- | ----------- | ------- |
| BUILD_DIR | Directory used by CMake to build OCCA | `./build` |
| INSTALL_DIR | Directory where OCCA should be installed | `./install` |
| BUILD_TYPE | Optimization and debug level | `RelWithDebInfo` |
| CXX | C++11 compiler | `g++` |
| CXXFLAGS | C++ compiler flags | *empty* | 
| CC | C11 compiler| `gcc` |
| CFLAGS | C compiler flags | *empty* |
| OCCA_ENABLE_CUDA | Enable use of the CUDA backend | `ON`|
| OCCA_ENABLE_HIP | Enable use of the HIP backend | `ON`|
| OCCA_ENABLE_DPCPP | Enable use of the DPC++ backend | `ON`|
| OCCA_ENABLE_OPENCL | Enable use of the OpenCL backend | `ON`|
| OCCA_ENABLE_OPENMP | Enable use of the OpenMP backend | `ON`|
| OCCA_ENABLE_METAL | Enable use of the Metal backend | `ON`|
| OCCA_ENABLE_TESTS | Build OCCA's test harness | `ON` |
| OCCA_ENABLE_EXAMPLES | Build OCCA examples | `ON` |
| OCCA_ENABLE_FORTRAN | Build the Fortran language bindings | `OFF`|
| OCCA_CLANG_BASED_TRANSPILER | Build clang based transpiler that support C++ in OKL | `OFF`|
| OCCA_LOCAL_CLANG_PATH | Set path to local clang dir for clang based transpiler | `STRING`|
| FC | Fortran 90 compiler | `gfortran` |
| FFLAGS | Fortran compiler flags | *empty* |

#### Dependency Paths


The following environment variables can be used to specify the path to third-party dependencies needed by different OCCA backends. The value assigned should be an absolute path to the parent directory, which typically contains subdirectories `bin`, `include`, and `lib`.

| Backend | Environment Variable | Description |
| --- | --- | --- |
| CUDA | CUDATookit_ROOT | Path to the CUDA the NVIDIA CUDA Toolkit |
| HIP | HIP_ROOT | Path to the AMD HIP toolkit |
| OpenCL | OpenCL_ROOT | Path to the OpenCL headers and library |
| DPC++ | SYCL_ROOT | Path to the SYCL headers and library |

### Building

After CMake configuration is complete, OCCA can be built with the command
```shell
$ cmake --build build --parallel <number-of-threads>
```

When cross compiling for a different platform, the targeted hardware doesn't need to be available; however all dependencies&mdash;e.g., headers, libraries&mdash;must be present. Commonly this is the case for large HPC systems, where code is compiled on login nodes and run on compute nodes.


#### Building with Clang transplier option


Hard dependency is clang-17 exactly. So far clang based transpiler does not have compatibility layer to support differences of C++ API in clang tooling in newer versions. How to install it please refer to the original 
[clang occa-transpiler](https://github.com/libocca/occa-transpiler/blob/main/README.md)

The rest dependencies are represented as git submodules and are fetched automatically by cmake script.
Building the project with clang based transpiler now is supported only by *CMake* build system.
All options must be provided directly. The following example shows how to build the new transpiler with system Clang:

```shell
$ mkdir build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release -DOCCA_CLANG_BASED_TRANSPILER=ON ..
$ cmake --build . --parallel <number-of-threads> 
$ cmake --install . --prefix install
```

For Clang that is built locally the install prefix should be specified:
```shell
$ mkdir build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release -DOCCA_CLANG_BASED_TRANSPILER=ON  -DOCCA_LOCAL_CLANG_PATH=/home/ikobein/rnd/projects/shell/okl-transpiler/clang-17-rel-instal..
$ cmake --build . --parallel <number-of-threads> 
$ cmake --install . --prefix install
```
### Testing

CTest is used for the OCCA test harness and can be run using the command
```shell
$ ctest --test-dir BUILD_DIR --output-on-failure
```

Before running CTest, it may be necessary to set the environment variables `OCCA_CXX` and `OCCA_CC` since OCCA defaults to using gcc and g++. Tests for some backends may return a false negative otherwise.

During testing, `BUILD_DIR/occa` is used for kernel caching. This directory may need to be cleared when rerunning tests after recompiling with an existing build directory.

### Installation

Commandline installation of OCCA can be accomplished with the following:
```shell
$ cmake --install BUILD_DIR --prefix INSTALL_DIR
```
During installation, the [Env Modules](Env_Modules) file `INSTALL_DIR/modulefiles/occa` is generated. When this module is loaded, paths to the installed `bin`, `lib`, and `include` directories are appended to environment variables such as `PATH` and `LD_LIBRARY_PATH`. 
To make use of this module, add the following to your `.modulerc` file
```
module use -a INSTALL_DIR/modulefiles
```
 then at the commandline call
```shell
$ module load occa
```

### Building an OCCA application

For convenience, OCCA provides CMake package files which are configured during installation. These package files define an imported target, `OCCA::libocca`, and look for all required dependencies.

For example, the CMakeLists.txt of downstream projects using OCCA would include
```cmake
find_package(OCCA REQUIRED)

add_executable(downstream-app ...)
target_link_libraries(downstream-app PRIVATE OCCA::libocca)

add_library(downstream-lib ...)
target_link_libraries(downstream-lib PUBLIC OCCA::libocca)
```
In the case of a downstream library, linking OCCA using the  `PUBLIC` specifier ensures that CMake will automatically forward OCCA's dependencies to applications which use the library.

## Mac OS

> Do you use OCCA on Mac OS? Help other Mac OS users by contributing to the documentation here!

## Windows

> Do you use OCCA on Windows? Help other Windows users by contributing to the documentation here!

[CMake]: https://cmake.org/
[Env_Modules]: https://modules.readthedocs.io/en/latest/index.html
