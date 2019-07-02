### Example: Arrays

We show the use of `occa::array<>` to simplifiy array operations along with auto syncs between host and device data

### Compiling the Example

```bash
make
```

### Usage

```
> ./main --help

Usage: ./main [OPTIONS]

Example using occa::array objects for easy allocation and host <-> device
syncing

Options:
  -d, --device     Device properties (default: "mode: 'Serial'")
  -h, --help       Print usage
  -v, --verbose    Compile kernels in verbose mode
```
