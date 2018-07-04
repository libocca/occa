<p align="center">
  <a href="https://libocca.org">
    <img alt="occa" src="https://libocca.org/assets/images/logo/blue.svg" width=250>
  </a>
</p>
&nbsp;
<p align="center">
  <a href="https://travis-ci.org/libocca/occa"><img alt="Build Status" src="https://travis-ci.org/libocca/occa.svg?branch=master"></a>
  <a href="https://codecov.io/github/libocca/occa"><img alt="codecov.io" src="https://codecov.io/github/libocca/occa/coverage.svg"></a>
  <a href="https://gitter.im/libocca/occa?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge"><img alt="Gitter" src="https://badges.gitter.im/libocca/occa.svg"></a>
</p>

&nbsp;

### What is OCCA?

In a nutshell, OCCA (like *oca*-rina) is an open-source library which aims to

- Make it easy to program different types of devices (e.g. _CPU_, _GPU_, _FPGA_)
- Provide a [unified API](https://libocca.org/#/guide/occa/introduction) for interacting with backend device APIs (e.g. _OpenMP_, _CUDA_, _OpenCL_)
- Use just-in-time compilation to build backend kernels
- Provide a [kernel language](https://libocca.org/#/guide/okl/introduction), a minor extension to C, to abstract programming for each backend

&nbsp;

### Links

- [Documentation](https://libocca.org)
- **Want to contribute?** Checkout the ['Good First Issue' issues](https://github.com/libocca/occa/issues?q=is%3Aopen+is%3Aissue+label%3A%22Good+First+Issue%22)
- **More of a challenge?** Checkout the ['Help Needed' issues](https://github.com/libocca/occa/issues?utf8=%E2%9C%93&q=is%3Aopen+is%3Aissue+label%3A%22Help+Wanted%22)
- 🌟 Who is using OCCA?
  - [Gallery](https://libocca.org/#/gallery)
  - [Publications](https://libocca.org/#/publications)

&nbsp;

### Installing

```bash
git clone --depth 1 https://github.com/libocca/occa.git
cd occa
make -j 4
```

&nbsp;

### Environment

Setup environment variables inside the `occa` directory

#### Linux

```bash
export PATH+=":${PWD}/bin"
export LD_LIBRARY_PATH+=":${PWD}/lib"
```

#### Mac OSX

```bash
export PATH+=":${PWD}/bin"
export DYLD_LIBRARY_PATH+=":${PWD}/lib"
```

&nbsp;

### Hello World

```bash
cd examples/1_add_vectors/cpp
make
./main
```

&nbsp;

### CLI

There is an executable `occa` provided inside `bin`

```bash
> occa --help

Usage: occa COMMAND

Can be used to display information of cache kernels.

Commands:
  autocomplete    Prints shell functions to autocomplete occa
                  commands and arguments
  cache           Cache kernels
  clear           Clears cached files and cache locks
  compile         Compile kernels
  env             Print environment variables used in OCCA
  info            Prints information about available backend modes
  modes           Prints available backend modes
  translate       Translate kernels
  version         Prints OCCA library version

Arguments:
  COMMAND    Command to run
```

&nbsp;

### Bash Autocomplete

```bash
. <(occa autocomplete bash)
```
