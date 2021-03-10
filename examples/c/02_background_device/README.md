# Example: Using the Background Device

This example shows how to avoid passing an `occa::device` object around by using the _background device_.

# Compiling the Example

```bash
make
```

## Usage

```
> ./main --help

Usage: ./main [OPTIONS]

Example showing how to use background devices, allowing passing of the device
implicitly

Options:
  -d, --device     Device properties (default: "{mode: 'Serial'}")
  -h, --help       Print usage
  -v, --verbose    Compile kernels in verbose mode
```
