# INSTALLATION GUIDE

## Requirements

### Minimum

 - CMake v3.21 or newer
 - C++17 compiler
 - C11 compiler

### Optional

 - Fortran 90 compiler
 - CUDA 9 or later
 - HIP 3.5 or later
 - SYCL 2020 or later
 - OpenCL 2.0 or later
 - OpenMP 4.0 or later
 - Support Clang based transpiler

## Configure, Build/Install and Test

OCCA uses CMake as the build system. For convenience, we also provide a few default
[cmake-presets](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html) in `CMakePresets.json` that users
can use and/or extend in order to customize the configure, build and test phases of OCCA.
OCCA provides default presets for each of the `build`, `configure` and `test` phases categorized by the most common
compilers (GNU, Clang, oneAPI, etc.).
A workflow preset is a combination of `configure`, `build` and `test` presets that lets users run all of these phases
sequentially together in a single command.
To list, the current `configure`, `build`, `test` and `workflow` presets, use `cmake --list-presets[=type]` command
as follows:
```bash
> cmake --list-presets=configure
Available configure presets:

  "system-default"
  "gnu-default"
  "clang-default"
  "oneapi-default"

> cmake --list-presets=workflow
Available workflow presets:

  "system-default" - Configure, build and install with system default compilers
  "gnu-default"    - Configure, build and install with GNU compilers
  "clang-default"  - Configure, build and install with LLVM/Clang compilers
  "oneapi-default" - Configure, build and install with oneAPI compilers
```

For example, to run `configure`, `build`, `install` and `test` phases using the GNU compilers, you can use the
`gnu-default` workflow preset as follows:
```bash
cmake --workflow --preset gnu-default
```
The above command is equal to the following three commands run sequentially one after the other:
```bash
cmake --preset gnu-default
cmake --build --preset gnu-default
ctest --preset gnu-default
```
The first command will configure the OCCA build based on the parameters defined in the preset. The
second step will build and install OCCA. Finally, the third step will run the test harness using `CTest`.
You will rarely has to run these steps in isolation. We recommend using the workflow presets in either
`CMakePresets.json` or creating a custom workflow as described in [Create Custom Presets](#create-custom-presets).

During installation, the [Env Modules](Env_Modules) file `OCCA_INSTALL_DIR/modulefiles/occa` is generated.
When this module is loaded, paths to the installed `bin`, `lib`, and `include` directories are appended to
environment variables such as `PATH` and `LD_LIBRARY_PATH`. 
To make use of this module, add the following to your `.modulerc` file
```bash
module use -a OCCA_INSTALL_DIR/modulefiles
```
 then at the commandline call
```bash
module load occa
```

**Note**: Before running CTest, it may be necessary to set the environment variables `OCCA_CXX` and `OCCA_CC`
since OCCA defaults to using gcc and g++. Tests for some backends may return a false negative otherwise.

**Note**: During testing, `OCCA_BUILD_DIR/occa` is used for kernel caching. This directory may need to be cleared
when rerunning tests after recompiling with an existing build directory.

### Create Custom Presets

`cmake-presets` can be modified by changing the `environment` section of the preset definition.
The following table list the build configurations which can be changed in the `environment` section in the
`configure`-type cmake-presets.

| Environment variable | Description | Default |
| --------- | ----------- | ------- |
| OCCA_BUILD_DIR | Directory used by CMake to build OCCA | `./build` |
| OCCA_BUILD_TYPE | Optimization and debug level | `RelWithDebInfo` |
| OCCA_INSTALL_DIR | Directory where OCCA should be installed | `./install` |
| CXX | C++11 compiler | *Depends on the preset* |
| CXXFLAGS | C++ compiler flags | *Depend on the preset* |
| CC | C11 compiler| *Depends on the preset* |
| CFLAGS | C compiler flags | *empty* |
| FC | Fortran 90 compiler | *Depends on the preset* |
| FFLAGS | Fortran compiler flags | *empty* |
| OCCA_ENABLE_OPENMP | Enable use of the OpenMP backend | `ON`|
| OCCA_ENABLE_OPENCL | Enable use of the OpenCL backend | `ON`|
| OCCA_ENABLE_DPCPP | Enable use of the DPC++ backend | `OFF`(`ON` for `oneapi-default`)|
| OCCA_ENABLE_CUDA | Enable use of the CUDA backend | `ON`|
| OCCA_ENABLE_HIP | Enable use of the HIP backend | `ON`|
| OCCA_ENABLE_METAL | Enable use of the Metal backend | `ON`|
| OCCA_ENABLE_FORTRAN | Build the Fortran language bindings | `ON`(`OFF` for `clang-default`) |
| OCCA_ENABLE_TESTS | Build OCCA test harness | `ON` |
| OCCA_ENABLE_EXAMPLES | Build OCCA examples | `ON` |
| OCCA_CLANG_BASED_TRANSPILER | Build clang based transpiler that support C++ in OKL | `OFF`|

Users can inherit the default presets in `CMakePresets.json` and then extend/change them based on their requirements.
In order to do so, edit `CMakeUserPresets.json` file to match your needs.
```json
{
  "version": 6,
  "cmakeMinimumRequired": {
    "major": 3,
    "minor": 21,
    "patch": 0
  },
  "configurePresets": [
    {
      "name": "my-local-config",
      "inherits": "gnu-default",
      "environment": {
        "CXXFLAGS": "-D_FORTIFY_SOURCE=2 -D_GLIBCXX_ASSERTIONS",
        "OCCA_ENABLE_METAL": "OFF",
        "OCCA_ENABLE_OPENMP": "OFF"
      }
    }
  ],
  "buildPresets": [
    {
      "name": "my-local-build",
      "inherits": "gnu-default",
      "configurePreset": "my-local-config",
      "jobs": 16
    }
  ],
  "testPresets": [
    {
      "name": "my-local-test",
      "inherits": "gnu-default",
      "configurePreset": "my-local-config"
    }
  ],
  "workflowPresets": [
    {
      "name": "my-local-workflow",
      "displayName": "Configure, build, install and test with on my local machine",
      "steps": [
        {
          "type": "configure",
          "name": "my-local-config"
        },
        {
          "type": "build",
          "name": "my-local-build"
        },
        {
          "type": "test",
          "name": "my-local-test"
        }
      ]
    }
  ]
}
```

In the above `CMakeUserPresets.json` file, we have created `my-local-config`, `my-local-build`, and `my-local-workflow`
by inheriting from the `gnu-default` workflow in `CMakePresets.json` file and updating a few configuration options.
Then you can run this custom preset using the following command:
```bash
cmake --workflow --preset my-local-workflow
```

### Backend Dependency Paths

The following variables can be used to specify the path to third-party dependencies needed by different OCCA backends.
The value assigned should be an absolute path to the parent directory, which typically contains subdirectories `bin`,
`include`, and `lib`.

| Backend | Environment Variable | Description |
| --- | --- | --- |
| CUDA | CUDATookit_ROOT | Path to the CUDA the NVIDIA CUDA Toolkit |
| HIP | HIP_ROOT | Path to the AMD HIP toolkit |
| OpenCL | OpenCL_ROOT | Path to the OpenCL headers and library |
| DPC++ | SYCL_ROOT | Path to the SYCL headers and library |

You can set these variables under the `cacheVariables` section of the preset. For example, you can change the
`my-local-config` from previous example as follows:
```json
{
  "version": 6,
  "cmakeMinimumRequired": {
    "major": 3,
    "minor": 21,
    "patch": 0
  },
  "configurePresets": [
    {
      "name": "my-local-config",
      "inherits": "gnu-default",
      "cacheVariables": {
        "CUDATookit_ROOT": "/usr/local/cuda-9.0"
      }
    }
  ]
}
```

### Building with Clang transpiler

occa-transpiler repository can be found in [libocca/occa-transpiler](https://github.com/libocca/occa-transpiler/).
Please refer [occa-transpiler README](https://github.com/libocca/occa-transpiler/blob/main/README.md) for instructions on
how to build and install the occa-transpiler.
Then you can modify the preset as following to install OCCA with occa-transpiler enabled.
Please replace `<occa-transpiler-install-dir>` by the root directory of your occa-transpiler installation.

```json
{
  "version": 6,
  "cmakeMinimumRequired": {
    "major": 3,
    "minor": 21,
    "patch": 0
  },
  "configurePresets": [
    {
      "name": "my-local-config",
      "inherits": "gnu-default",
      "environment": {
        "OCCA_CLANG_BASED_TRANSPILER": "ON"
      },
      "cacheVariables": {
        "CMAKE_PREFIX_PATH": "<occa-transpiler-install-dir>"
      }
    }
  ]
}
```

## Building an OCCA application

For convenience, OCCA provides CMake package files which are configured during installation. These package files define
an imported target, `OCCA::libocca`, and look for all required dependencies.

For example, the CMakeLists.txt of downstream projects using OCCA would include
```cmake
find_package(OCCA REQUIRED)

add_executable(downstream-app ...)
target_link_libraries(downstream-app PRIVATE OCCA::libocca)

add_library(downstream-lib ...)
target_link_libraries(downstream-lib PRIVATE OCCA::libocca)
```
In the case of a downstream library, linking OCCA using the  `PUBLIC` specifier ensures that CMake will automatically
forward OCCA's dependencies to applications which use the library.

## Mac OS

> Do you use OCCA on Mac OS? Help other Mac OS users by contributing to the documentation here!

## Windows

> Do you use OCCA on Windows? Help other Windows users by contributing to the documentation here!

[CMake]: https://cmake.org/
[Env_Modules]: https://modules.readthedocs.io/en/latest/index.html
