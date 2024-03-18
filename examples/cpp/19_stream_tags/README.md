# Example: Events

GPU devices introduce `streams`, which potentially allow parallel queueing of instructions

`Stream tags` are used to query and manage (synchronize) those streams

This example shows how to setup `occa::streamTag` to manage jobs in different streams

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
