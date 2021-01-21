# Example: Unified Memory

We show unified memory which automatically syncs data between host and device

Transfers are dome between kernel launches and device synchronization (`occa::device::finish()`) when needed

# Compiling the Example

```bash
make
```

## Usage

```
> ./main --help

Usage: ./main [OPTIONS]

Example using unified memory, where host and device data is mirrored and synced

Options:
  -d, --device     Device properties (default: "{mode: 'Serial'}")
  -h, --help       Print usage
  -v, --verbose    Compile kernels in verbose mode
```
