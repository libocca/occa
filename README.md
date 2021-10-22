<p align="center">
  <a href="https://libocca.org">
    <img alt="occa" src="https://libocca.org/assets/images/logo/blue.svg" width=250>
  </a>
</p>
&nbsp;

<div align="center"> 

[![license](https://img.shields.io/github/license/libocca/occa)](LICENSE)
[![slack](https://img.shields.io/badge/Chat-on%20Slack-%23522653)][OCCA_SLACK]
![github-ci](https://github.com/libocca/occa/workflows/Build/badge.svg)
![codecov](https://codecov.io/github/libocca/occa/coverage.svg)
</div>

&nbsp;

## Performance, Portability, Transparency

OCCA is an open source, portable, and vendor neutral framework for parallel programming on heterogeneous platforms. The OCCA API provides unified models for—such as a device, memory, or kernel—which are common to other programming models. The OCCA Kernel Language (OKL) enables the creation of portable device kernels using a directive-based extension to the C-language. 

Mission critical computational science and engineering applications from the public and private sectors rely on OCCA. Notable users include the U.S. Department of Energy and Royal Dutch Shell.

**Key Features**

- Muitiple backends&mdash; including CUDA, HIP, Data Parallel C++, OpenCL, OpenMP, and Metal
- JIT compilation and caching of kernels
- Language support for C, C++, and Fortran
- Interoperability with backend API and kernels
- Transparency **...**


## Requirements

- [CMake] v3.17 or newer
- C++11 compliant compiler
- ...

Details for requirements specific to each backend can be found in the [installation guide](INSTALL.md)

## Build, Test, Install

OCCA uses the [CMake] build system. For a comprehensive overview of all build settings, checkout the [installation guide](INSTALL.md)

### Linux 

For conveinence, the shell script `configure.sh` has been provided drive the Cmake build. Compilers, flags, and other build parameters can be adjusted there. By default OCCA will be built and installed in `./build` and `./install`.

The following sequence of commands demonstrates a typical sequence of shell commands to build, test, and install occa:
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

OCCA provides CMake package files which are included during installation. To use OCCA in a downstream project

### Environment

...

## Community

### Support

Need help? Checkout the documentation at libocca.org or ask a question in the \#help channel on [Slack][OCCA_SLACK]

### Feedback

To provide feedback, start a conversation on [Slack][OCCA_SLACK] on the \#general or \#ideas channel.

### Get Involved
OCCA is a community driven project that relies on the support of people like you! For ways to get involved, see our [contributing guidelines](CONTRIBUTING.md).

### Acknowledgements

> This work was supported by Argonne Leadership Computing Facility, which is a DOE Office of Science User Facility supported under Contract DE-AC02-06CH11357 and by the Exascale Computing Project (17-SC-20-SC), a joint project of the U.S. Department of Energy’s Office of Science and National Nuclear Security Administration, responsible for delivering a capable exascale ecosystem, including software, applications, and hardware technology, to support the nation’s exascale computing imperative.

## License

OCCA is available under a [MIT license](LICENSE.MD)

## Similar Projects

- [Alpaka](https://github.com/alpaka-group/alpaka)

  > The alpaka library is a header-only C++14 abstraction library for accelerator development. Its aim is to provide performance portability across accelerators through the abstraction (not hiding!) of the underlying levels of parallelism.

- [RAJA](https://github.com/LLNL/RAJA)

   > RAJA is a library of C++ software abstractions, primarily developed at Lawrence Livermore National Laboratory (LLNL), that enables architecture and programming model portability for HPC applications

- [Kokkos](https://github.com/kokkos/kokkos)

   > Kokkos Core implements a programming model in C++ for writing performance portable applications targeting all major HPC platforms. For that purpose it provides abstractions for both parallel execution of code and data management.

[OCCA_SLACK]: https://join.slack.com/t/libocca/shared_invite/zt-4jcnu451-qPpPWUzhm7YQKY_HMhIsIw

[CMake]: https://cmake.org/