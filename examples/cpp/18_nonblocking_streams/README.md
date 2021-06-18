# Example: Non-blocking Streams

GPU devices introduce `streams`, which potentially allow parallel queueing of instructions

Especially with non-blocking streams created operations in those streams will not have implicit synchronizations with the default stream

This example shows how to setup `occa::streams` with the non-blocking property

# Compiling the Example

```bash
make
```

## Usage

```
> ./main --help

Usage: ./main [OPTIONS]

Example showing the use of multiple non-blocking streams in a device

Options:
  -d, --device     Device properties (default: "{mode: 'CUDA', device_id: 0}")
  -h, --help       Print usage
  -v, --verbose    Compile kernels in verbose mode
```
