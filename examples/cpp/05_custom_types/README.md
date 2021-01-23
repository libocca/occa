# Example: Runtime Type Checking

Example with custom dtypes, showcasing runtime type checking

- Create basic, struct, and dtuple dtypes

## Explanation for dtypes

occa::memory can have a dtype to enable runtime type checking

`dtype_t` used in typed memory allocations must be 'global'

Global `dtype_t` objects are treated as singletons and assumed to exist while the memory objects are still alive

NOTE:

- Don't deallocate used dtype_t
- Don't use local dtype_t objects

# Compiling the Example

```bash
make
```

## Usage

```
> ./main --help

Usage: ./main [OPTIONS]

Example with custom dtypes

Options:
  -d, --device     Device properties (default: "{mode: 'Serial'}")
  -h, --help       Print usage
  -v, --verbose    Compile kernels in verbose mode
```
