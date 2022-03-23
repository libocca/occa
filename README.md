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
- C++11 compiler
- C11 compiler

### Optional

 - Fortan 90 compiler
 - MPI 2.0+
 - CUDA vXXX
 - HIP vXXX
 - oneAPI Toolkit vXXX
 - OpenCL ...
 - OpenMP ...


A detailed list of tested platforms can be found in the [installation guide](INSTALL.md).


## Build, Test, Install

OCCA uses the [CMake] build system. Checkout the [installation guide](INSTALL.md) for a comprehensive overview of all build settings.

### Linux 

For convenience, the shell script `configure.sh` has been provided drive the Cmake build. Compilers, flags, and other build parameters can be adjusted there. By default OCCA will be built and installed in `./build` and `./install`.

The following demonstrates a typical sequence of shell commands to build, test, and install occa:
```
$> ./configure.sh
$> cmake --build build --parallel <number-of-threads>
$> ctest --test-dir build --output-on-failure
$> cmake --install build --prefix install
```

### MacOS

...

### Windows

...

## Use

### Building an OCCA application

OCCA provides CMake package files which are included during installation. 
To use OCCA in a downstream project (...)

### Environment

...

## Community

### Support

Need help? Checkout the [repository wiki](https://github.com/libocca/occa/wiki) or ask a question in the [Q&A discussions category](https://github.com/libocca/occa/discussions/categories/q-a).

### Feedback

To provide feedback, start a conversation in the [general](https://github.com/libocca/occa/discussions/categories/general) or [ideas](https://github.com/libocca/occa/discussions/categories/ideas) discussion categories.

### Get Involved
OCCA is a community driven project that relies on the support of people like you! For ways to get involved, see our [contributing guidelines](CONTRIBUTING.md).

### Acknowledgements

> This work was supported by Argonne Leadership Computing Facility, which is a DOE Office of Science User Facility supported under Contract DE-AC02-06CH11357 and by the Exascale Computing Project (17-SC-20-SC), a joint project of the U.S. Department of Energy’s Office of Science and National Nuclear Security Administration, responsible for delivering a capable exascale ecosystem, including software, applications, and hardware technology, to support the nation’s exascale computing imperative.

## License

OCCA is available under a [MIT license](LICENSE.MD)


[OCCA_WEBSITE]: https://libocca.org

[OCCA_SLACK]: https://join.slack.com/t/libocca/shared_invite/zt-4jcnu451-qPpPWUzhm7YQKY_HMhIsIw

[CMake]: https://cmake.org/