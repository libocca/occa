# Example: Arrays

Example using `occa::forLoop` for inline kernels

## Description

Similar to `occa::array`, `occa::forLoop` allows us to build for-loop kernels inline. This allows injecting arguments along with adding compile-time defines.

```cpp
  occa::scope scope({
    {"output", output}
  }, {
    {"defines/length", length}
  });

  occa::forLoop()
    .outer(2)
    .inner(length)
    .run(OCCA_FUNCTION(scope, [=](const int outerIndex, const int innerIndex) -> void {
      const int globalIndex = outerIndex + (2 * innerIndex);
      output[globalIndex] = globalIndex;
    }));
```

### Iterators

`.outer()` and `.inner()` support 1, 2, or 3 arguments which can be of types:
- `int N` which generates a loop between `[0, N)`
- `occa::range` which generates a loop given the range `start`, `end`, and `step` definition
- `occa::array<int>` which iterates through the indices of the array

### For-loop body

Based on the `.outer` and `.inner` argument counts, the for-loop body will expect a lambda with the proper types

A few examples:
- `.outer(N)` -> `[=](const int outerIndex) -> void {}`
- `.outer(N, N)` -> `[=](const int2 outerIndex) -> void {}`
- `.outer(N, N).inner(N)` -> `[=](const int2 outerIndex, const int innerIndex) -> void {}`
- `.outer(N).inner(N, N)` -> `[=](const int outerIndex, const int2 innerIndex) -> void {}`

### @.outer-only loops

The use of `@shared` memory can be crucial for some implementations. Because of this, we easily support `@shared` memory  by automating only the `@outer` loop generation and leaving the `@inner` loop implementations to the user.

Note the weird usage of `OKL("...");`. This is used to inject source-code due to compiler restrictions:
1. The lambda is actually compiled, so it must be valid C++11. This means OKL attributes can't be placed inline.
2. The source-code is loaded by stringifying the lambda. Unfortunately the preprocessor doesn't keep the newlines so `OKL(<source-code>)` can be used to bypass this issue. This is useful to setup directives, such as `#if` / `#endif`.

```cpp
  occa::forLoop()
    .outer(length)
    .run(OCCA_FUNCTION(scope, [=](const int outerIndex) -> void {
      OKL("@shared"); int array[2];

      OKL("@inner");
      for (int i = 0; i < 2; ++i) {
        array[i] = i;
      }

      OKL("@inner");
      for (int i = 0; i < 2; ++i) {
        output[i] = array[1 - i];
      }
    }));
```

# Compiling the Example

```bash
make
```

## Usage

```
> ./main --help

Usage: ./main [OPTIONS]

Example using occa::forLoop for inline kernels

Options:
  -d, --device     Device properties (default: "{mode: 'Serial'}")
  -h, --help       Print usage
  -v, --verbose    Compile kernels in verbose mode
```
