### Example: Building Kernels from Strings

A simple example to show the flexibility of JIT compilation by producing the kernel source on the fly

### Compiling the Example

```bash
make
```

### Usage

```
> ./main --help

Usage: ./main [OPTIONS]

Example which shows run-time kernel source code generation

Options:
  -d, --device     Device properties (default: "mode: 'Serial'")
  -h, --help       Print usage
  -v, --verbose    Compile kernels in verbose mode
```
