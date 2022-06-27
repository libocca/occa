# Example: Add Vectors

A 'Hello World' example showing the basics

- Creating an OCCA device
- Allocating and setting memory
- Building a kernel (function that runs on the device)

# Compiling the Example

```bash
make
```

## Usage

```
> ./main --help

Usage: ./main [OPTIONS]

Example adding two vectors

Options:
  -d, --device     Device properties (default: "{mode: 'Serial'}")
  -h, --help       Print usage
  -v, --verbose    Compile kernels in verbose mode
```
