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

### Table of Contents

- [What is OCCA?](#what-is-occa)
- [Documentation](#documentation)
- [How to build](#build)
  - [Linux](#build-linux)
  - [MacOS](#build-macos)
- [Examples](#examples)
  - [Hello World](#examples-hello-world)
  - [Inline for-loops](#examples-for-loops)
  - [Arrays + Functional Programming](#examples-arrays)
- [CLI](#cli)
  - [Bash Autocomplete](#cli-autocomplete)
- [Similar Libraries](#similar-libraries)

&nbsp;

<h2 id="what-is-occa">What is OCCA?</h2>

In a nutshell, OCCA (like *oca*-rina) is an open-source library which aims to

- Make it easy to program different types of devices (e.g. _CPU_, _GPU_, _FPGA_)
- Provide a [unified API](https://libocca.org/#/guide/occa/introduction) for interacting with backend device APIs (e.g. _OpenMP_, _CUDA_, _HIP_, _OpenCL_, _Metal_)
- JIT compile backend kernels and provide a [kernel language](https://libocca.org/#/guide/okl/introduction) (a minor extension to C) to abstract programming for each backend

The "Hello World" example of adding two vectors looks like:

```cpp
@kernel void addVectors(const int entries,
                        const float *a,
                        const float *b,
                        float *ab) {
  for (int i = 0; i < entries; ++i; @tile(16, @outer, @inner)) {
    ab[i] = a[i] + b[i];
  }
}
```

Or we can inline it using C++ lambdas

```cpp
// Capture variables
occa::scope scope({
  {"a", a},
  {"b", b},
  {"ab", ab}
});

occa::forLoop()
  .tile({entries, 16})
  .run(OCCA_FUNCTION(scope, [=](const int i) -> void {
    ab[i] = a[i] + b[i];
  }));
```

Or we can use a more functional way by using `occa::array`

```cpp
// Capture variables
occa::scope scope({
  {"b", b}
});

occa::array<float> ab = (
  a.map(OCCA_FUNCTION(
    scope,
    [=](const float &value, const int index) -> float {
      return value + b[index];
    }
  ))
);
```

&nbsp;

<h2 id="documentation">Documentation</h2>

We maintain our documentation on the [libocca.org](https://libocca.org) site

- [:notebook:	Guide](https://libocca.org/#/guide)
- [:gear: API](https://libocca.org/#/api/)
- [🌟 Who is using OCCA?](https://libocca.org/#/gallery)
- [:lab_coat: Publications](https://libocca.org/#/publications)

&nbsp;

<h2 id="build">How to build</h2>

```bash
git clone --depth 1 https://github.com/libocca/occa.git
cd occa
make -j 4
```

Setup environment variables inside the `occa` directory

<h3 id="build-linux">Linux</h2>

```bash
export PATH+=":${PWD}/bin"
export LD_LIBRARY_PATH+=":${PWD}/lib"
```

<h3 id="build-macos">MacOS</h2>

```bash
export PATH+=":${PWD}/bin"
export DYLD_LIBRARY_PATH+=":${PWD}/lib"
```

&nbsp;

<h2 id="examples">Examples</h2>

<h3 id="examples-hello-world">Hello World</h3>

The occa library is based on 3 different objects, all covered in the [01_add_vectors](/examples/cpp/01_add_vectors) example:
- `occa::device`
- `occa::memory`
- `occa::kernel`

```bash
cd examples/cpp/01_add_vectors
make
./main
```

<h3 id="examples-for-loops">Inline for-loops</h3>

Find how to inline `for` loops using `occa::forLoop` in example [02_for_loops](/examples/cpp/02_for_loops):

```bash
cd examples/cpp/02_for_loops
make
./main
```

&nbsp;

<h3 id="examples-arrays">Arrays + Functional Programming</h3>

Learn how to use `occa::array` in a functional way in example [03_arrays](/examples/cpp/03_arrays):

```bash
cd examples/cpp/03_arrays
make
./main
```

&nbsp;

<h2 id="cli">CLI</h2>

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

<h3 id="cli-autocomplete">Bash Autocomplete</h3>

```bash
if which occa > /dev/null 2>&1; then
    eval "$(occa autocomplete bash)"
fi
```

<h2 id="similar-libraries">Similar Libraries</h2>

OCCA is definitely not the only solution that aims to simplify programming on different hardware/accelerators.
Here is a list of other libraries that have taken different approaches:

- [Alpaka](https://github.com/alpaka-group/alpaka)

  > The alpaka library is a header-only C++14 abstraction library for accelerator development. Its aim is to provide performance portability across accelerators through the abstraction (not hiding!) of the underlying levels of parallelism.

- [RAJA](https://github.com/LLNL/RAJA)

   > RAJA is a library of C++ software abstractions, primarily developed at Lawrence Livermore National Laboratory (LLNL), that enables architecture and programming model portability for HPC applications

- [Kokkos](https://github.com/kokkos/kokkos)

   > Kokkos Core implements a programming model in C++ for writing performance portable applications targeting all major HPC platforms. For that purpose it provides abstractions for both parallel execution of code and data management.
