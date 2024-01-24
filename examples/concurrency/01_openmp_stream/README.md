# Example: Concurrency: OpenMP stream

The concept of stream is implemented in OpenMP mode using its threads.

This example shows launching kernels on multiple streams exact same way done on other modes that provides implementation of stream on their platform.

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
  -d, --device     Device properties (default: "{mode: 'OpenMP'}")
  -h, --help       Print usage
  -v, --verbose    Compile kernels in verbose mode
```
