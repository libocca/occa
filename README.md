<p align="center">
  <a href="https://libocca.org">
    <img alt="occa" src="https://libocca.org/assets/images/logo/blue.svg" width=250>
  </a>
</p>
&nbsp;

<div align="center"> 

[![license](https://img.shields.io/github/license/libocca/occa)](LICENSE)
![discussions](https://img.shields.io/github/discussions/libocca/occa)
[![slack](https://img.shields.io/badge/Chat-on%20Slack-%23522653)][OCCA_SLACK]
![github-ci](https://github.com/libocca/occa/workflows/Build/badge.svg)
![codecov](https://codecov.io/github/libocca/occa/coverage.svg)
[![twitter](https://img.shields.io/twitter/url?label=Twitter&style=social&url=https%3A%2F%2Ftwitter.com%2Flibocca)](https://twitter.com/libocca)
</div>

&nbsp;

## Performance, Portability, Transparency

OCCA is an open source, portable, and vendor neutral framework for parallel programming on heterogeneous platforms. The OCCA API provides unified models for concepts—such as a device, memory, or kernel—which are common to other programming models. The OCCA Kernel Language (OKL) enables the creation of portable device kernels using a directive-based extension to the C-language. 

Mission critical computational science and engineering applications from the public and private sectors rely on OCCA. Notable users include the U.S. Department of Energy and Royal Dutch Shell.

**Key Features**

- **Muitiple backends**&mdash; including CUDA, HIP, Data Parallel C++, OpenCL, OpenMP, and Metal
- **JIT compilation** and caching of kernels
- C, C++, and ***Fortran*** language support
- **Interoperability** with backend API and kernels
- **Transparency**&mdash;easy to understand how your code is mapped to each platform


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
 - OpenMP 4.0 or later


A detailed list of tested platforms can be found in the [installation guide](INSTALL.md).


## Build, Test, Install

OCCA uses the [CMake] build system. Checkout the [installation guide](INSTALL.md) for a comprehensive overview of all build settings and instructions for building on [Windows](INSTALL.md#windows) or [Mac OS](INSTALL.md#mac-os). 

### Linux 

For convenience, the shell script `configure-cmake.sh` has been provided to drive the CMake build. Compilers, flags, and other build parameters can be adjusted there. By default, this script uses `./build` and `./install` for the build and install directories.

The following demonstrates a typical sequence of shell commands to build, test, and install occa:
```shell
$ ./configure.sh
$ cmake --build build --parallel <number-of-threads>
$ ctest --test-dir build --output-on-failure
$ cmake --install build --prefix install
```

If dependencies are installed in a non-standard location, set the corresponding [environment variable](INSTALL.md#dependency-paths) to this path. 


## Use

### Building an OCCA application

For convenience, OCCA provides CMake package files which are configured during installation. These package files define an imported target, `OCCA::libocca`, and look for all required dependencies.

For example, the CMakeLists.txt of downstream projects using OCCA would include
```cmake
find_package(occa REQUIRED)

add_executable(downstream-app ...)
target_link_libraries(<downstream-app> PRIVATE OCCA::libocca)

add_library(downstream-lib ...)
target_link_libraries(<downstream-lib> PUBLIC OCCA::libocca)
```
In the case of a downstream library, linking OCCA using the  `PUBLIC` specifier ensures that CMake will forward OCCA's dependencies and compiler flags automatically.

### Environment

During installation, the [Env Modules](Env_Modules) file `<install-prefix>/modulefiles/occa` is generated. When this module is loaded, paths to the installed `bin`, `lib`, and `include` directories are appended to environment variables such as `LD_LIBRARY_PATH`.

## Community

### Support

Need help? Checkout the [repository wiki](https://github.com/libocca/occa/wiki) or ask a question in the [Q&A discussions category](https://github.com/libocca/occa/discussions/categories/q-a).

### Feedback

To provide feedback, start a conversation in the [general](https://github.com/libocca/occa/discussions/categories/general) or [ideas](https://github.com/libocca/occa/discussions/categories/ideas) discussion categories.

### Get Involved
OCCA is a community driven project that relies on the support of people like you! For ways to get involved, see our [contributing guidelines](CONTRIBUTING.md).

### Acknowledgements

This work was supported in part by 
- Argonne Leadership Computing Facility, which is a DOE Office of Science User Facility supported under Contract DE-AC02-06CH11357
- The Exascale Computing Project (17-SC-20-SC), a joint project of the U.S. Department of Energy’s Office of Science and National Nuclear Security Administration, responsible for delivering a capable exascale ecosystem, including software, applications, and hardware technology, to support the nation’s exascale computing imperative
- The Center for Efficient Exascale Discretizations (CEED), a co-design center within the U.S. Department of Energy Exascale Computing Project.
- Intel
- AMD
- Shell



## License

OCCA is available under a [MIT license](LICENSE.MD)


[OCCA_WEBSITE]: https://libocca.org

[OCCA_SLACK]: https://join.slack.com/t/libocca/shared_invite/zt-4jcnu451-qPpPWUzhm7YQKY_HMhIsIw

[CMake]: https://cmake.org/

[Env_Modules]: https://modules.readthedocs.io/en/latest/index.html