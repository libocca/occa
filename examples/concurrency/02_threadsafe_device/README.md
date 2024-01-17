# Example: Concurrency: threadsafe block

One of the challenges implementing concurrency is to make sure that OCCA properties desired for an operation excuted by one thread do not change by any other.

This example shows an API in OCCA to create a thread-safe block that only the current thread can modify OCCA properties such as a stream set for the current device.

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
