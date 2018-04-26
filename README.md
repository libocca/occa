<p align="center">
  <a href="https://libocca.org">
    <img alt="occa" src="./docs/_images/blue-logo.svg" width=250>
  </a>
</p>

<p align="center">
  <a href="https://travis-ci.org/libocca/occa"><img alt="Build Status" src="https://travis-ci.org/libocca/occa.svg"></a>
  <a href="https://codecov.io/github/libocca/occa"><img alt="codecov.io" src="https://codecov.io/github/libocca/occa/coverage.svg"></a>
  <a href="https://gitter.im/libocca/occa?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge"><img alt="Gitter" src="https://badges.gitter.im/libocca/occa.svg"></a>
</p>

---

OCCA is an open-source (MIT license) library used to program current multi-core/many-core architectures.
Devices (such as CPUs, GPUs, Intel's Xeon Phi, FPGAs, etc) are abstracted using an offload-model for application development and programming for the devices is done through a C-based (OKL) kernel.
OCCA gives developers the ability to target devices at run-time by using run-time compilation for device kernels.

## Links

* [Documentation](https://libocca.org)

## Installing

```bash
git clone https://github.com/libocca/occa.git
cd occa
make -j 4
```

## Environment

Setup environment variables inside the `occa` directory

### Linux

```bash
export PATH+=":${PWD}/bin"
export LD_LIBRARY_PATH+=":${PWD}/lib"
```

## Mac OSX

```bash
export PATH+=":${PWD}/bin"
export DYLD_LIBRARY_PATH+=":${PWD}/lib"
```

## Hello World

```bash
cd examples/1_add_vectors/cpp
make
./main
```

## CLI

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
  env             Print environment variables used in OCCA
  info            Prints information about available OCCA modes
  version         Prints OCCA library version

Arguments:
  COMMAND    Command to run
```

## Bash Autocomplete

```bash
. <(occa autocomplete bash)
```
