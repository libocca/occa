# Example: Streams

GPU devices introduce `streams`, which potentially allow parallel queueing of instructions

This example shows how to setup `occa::streams` which mirror GPU `streams`

# Compiling the Example

```bash
make
```

## Usage

```
> ./main --help

Usage: ./main [OPTIONS]

Example showing the use of multiple device streams

Options:
  -d, --device     Device properties (default: "{mode: 'Serial'}")
  -h, --help       Print usage
  -v, --verbose    Compile kernels in verbose mode
```
