<p align="center">
  <a href="https://libocca.org">
    <img alt="occa" src="https://libocca.org/assets/images/logo/blue.svg" width=250>
  </a>
</p>
&nbsp;
<p align="center">
  <a href="https://github.com/libocca/occa/workflows/Build/badge.svg"><img alt="Build" src="https://github.com/libocca/occa/workflows/Build/badge.svg"></a>
  <a href="https://codecov.io/github/libocca/occa"><img alt="codecov.io" src="https://codecov.io/github/libocca/occa/coverage.svg"></a>
  <a href="https://join.slack.com/t/libocca/shared_invite/zt-4jcnu451-qPpPWUzhm7YQKY_HMhIsIw"><img alt="Slack" src="https://img.shields.io/badge/Chat-on%20Slack-%23522653?logo=slack"></a>
</p>

&nbsp;

### What is OCCA?

In a nutshell, OCCA (like *oca*-rina) is an open-source library which aims to

- Make it easy to program different types of devices (e.g. _CPU_, _GPU_, _FPGA_)
- Provide a [unified API](https://libocca.org/#/guide/occa/introduction) for interacting with backend device APIs (e.g. _OpenMP_, _CUDA_, _HIP_, _OpenCL_, _Metal_)
- Use just-in-time compilation to build backend kernels
- Provide a [kernel language](https://libocca.org/#/guide/okl/introduction), a minor extension to C, to abstract programming for each backend

&nbsp;

### Links

- [Documentation](https://libocca.org)
- **Want to contribute?** Checkout the ['beginner' issues](https://github.com/libocca/occa/labels/beginner)
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
cd examples/cpp/1_add_vectors
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
