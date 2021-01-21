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
  - [:notebook:	Guide](https://libocca.org/#/guide)
  - [:gear: API](https://libocca.org/#/api)
  - [🌟 Who is using OCCA?](https://libocca.org/#/gallery)
  - [:lab_coat: Publications](https://libocca.org/#/publications)

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

### Examples

#### Basics

The occa library is based on 3 different objects, all covered in the [01_add_vectors](/examples/cpp/01_add_vectors) example:
- `occa::device`
- `occa::memory`
- `occa::kernel`

```bash
cd examples/cpp/01_add_vectors
make
./main
```

#### Functional Programming + Arrays

Learn how to use `occa::array` in a functional way in example [05_arrays](/examples/cpp/05_arrays):

```bash
cd examples/cpp/05_arrays
make
./main
```

&nbsp;

### CLI

There is an executable `occa` provided inside `bin`

```bash
> occa

Usage: occa [OPTIONS] COMMAND [COMMAND...]

Helpful utilities related to OCCA workflows

Commands:
  autocomplete    Prints shell functions to autocomplete occa
                  commands and arguments
  clear           Clears cached files and cache locks
  compile         Compile kernels
  env             Print environment variables used in OCCA
  info            Prints information about available backend modes
  modes           Prints available backend modes
  translate       Translate kernels
  version         Prints OCCA version

Arguments:
  COMMAND    Command to run

Options:
  -h, --help    Print usage
```

&nbsp;

### Bash Autocomplete

```bash
if which occa > /dev/null 2>&1; then
    eval "$(occa autocomplete bash)"
fi
```
