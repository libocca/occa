# Example: Arrays

Example using `occa::array` with inline lambdas

Updates the `occa::array` to be a type-safe wrapper on `occa::memory` with flexible functional methods.

## occa::function

We introduce a preprocessor macro called `OCCA_FUNCTION` that takes a C++ lambda and returns a typed `occa::function`!

**Example**

```cpp
occa::function<bool(int, int, const int*)> func = (
  OCCA_FUNCTION({}, [](const int entry, const int index, const int *entries) -> bool {
    return false;
  })
);
```

The `occa::function` template definition is taken from `std::function`, where a user can define the return and argument types. It's also callable just like an `std::function`

```cpp
func(0, 0, NULL);
```

## occa::array

We introduce a simple wrapper on `occa::memory` which is typed and contains some of the core `map` and `reduce` functional methods.

**Example**

```cpp
const double dynamicValue = 10;
const double compileTimeValue = 100;

occa::scope scope({
  // Passed as arguments
    {"dynamicValue", dynamicValue}
  }, {
  // Passed as compile-time #defines
    {"defines/compileTimeValue", compileTimeValue}
});

occa::array<double> doubleValues = (
  values.map(OCCA_FUNCTION(scope, [](int value) -> double {
    return compileTimeValue + (dynamicValue * value);
  }));
);
```

### Core methods
- `forEach`
- `mapTo`
- `map`
- `reduce`

### Reduction
- `dot`
- `every`
- `max`
- `min`
- `some`

### Re-indexing
- `reverse`
- `shiftLeft`
- `shiftRight`

### Utility
- `cast`
- `clamp`
- `clampMax`
- `clampMin`
- `concat`
- `fill`
- `slice`

### Search
- `findIndex`
- `find`
- `includes`
- `indexOf`
- `lastIndexOf`

# Compiling the Example

```bash
make
```

## Usage

```
> ./main --help

Usage: ./main [OPTIONS]

Example using occa::array with inline lambdas

Options:
  -d, --device     Device properties (default: "{mode: 'Serial'}")
  -h, --help       Print usage
  -v, --verbose    Compile kernels in verbose mode
```
