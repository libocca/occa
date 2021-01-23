# Example: Reduction

We show a more complex kernel which computes a reduction on a vector (adds all the entries up)

In this example, we show how `@shared` memory is used along with multiple `@inner` loops for synchronizing reads and writes to `@shared` memory

# Compiling the Example

```bash
make
```

## Usage

```
> ./main --help

Usage: ./main [OPTIONS]

Example of a reduction kernel which sums a vector in parallel

Options:
  -d, --device     Device properties (default: "{mode: 'Serial'}")
  -h, --help       Print usage
  -v, --verbose    Compile kernels in verbose mode
```
