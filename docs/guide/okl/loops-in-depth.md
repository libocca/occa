# Loops in Depth

A goal for OKL is to try and hide as much _magic_ from the user (what you see is what you get).

Unfortunately, _magic_ is introduced once we enter the outer loops since we execute that code in a device.

In order for OKL to work on supported backends, we place a few restrictions when dealing with outer and inner loops.

## Multiple Outer Loops

A kernel can have multiple outer loops which, similar to inner loops, guarantee all work is done before starting the second set of outer loops

For example:

```okl
for (int i = 0; i < N; ++i; @outer) {
  a[i] = i;
}
for (int i = 0; i < N; ++i; @outer) {
  b[i] = a[N - i - 1];
}
```

## Multiple Inner Loops

Similar to outer loops, writing multiple inner loops is completely legal.
**However**, there is a restriction that all inner loops must have the same iteration count.

#### <span class="correct">Correct</span>

```okl
for (...; @outer) {
  for (int i = 0; i < 10; ++i; @inner) {
  }
  for (int i = 10; i < 20; ++i; @inner) {
  }
}
```

#### <span class="incorrect">Incorrect</span>

```okl
for (...; @outer) {
  for (int i = 0; i < 10; ++i; @inner) {
  }
  for (int i = 0; i < 20; ++i; @inner) {
  }
}
```

### Future Plans
We plan on making it simpler for users in the future, such as

```okl
for (...; @outer) {
  @safeInner(N, M) {
    for (int i = 0; i < N; ++i; @inner) {
    }
    for (int i = 0; i < M; ++i; @inner) {
    }
  }
}
```

<md-icon class="transform-arrow">arrow_downward</md-icon>

```okl
for (...; @outer) {
  const int innerSize = max(N, M);
  for (int i = 0; i < innerSize; ++i; @inner) {
    if (i < N) {
    }
  }
  for (int i = 0; i < innerSize; ++i; @inner) {
    if (i < M) {
    }
  }
}
```

## Using Host inside Kernel

To remove some of the _magic_ inside OKL, here's an explicit list of what runs on the _host_ and what runs on the _device_ inside a `@kernel`

<span style="font-size: 1.1em">_Compute_</span>
<template><div style="margin-top: -0.8em; padding-left: 1em;">
**Host**: Used for computing everything outside of outer loops, including calculating outer loop bounds
<br />
**Device**: Computes outer loops
</div></template>

<span style="font-size: 1.1em">_Arguments_</span>
<template><div style="margin-top: -0.8em; padding-left: 1em;">
**Host**: Can use non-pointer arguments for computation (_Good_&nbsp;: &nbsp; `const int N`, &nbsp; _Bad_&nbsp;: &nbsp; `const int *array`)
<br />
**Device**: Can use all arguments for computation
</div></template>

Few examples include

#### <span class="incorrect">Incorrect</span>

```okl
@kernel void myKernel(const int *N) {
  const int N2 = 2 * (*N); // [Error] Can't access device memory
  for (int i = 0; i < N2; ++i; @outer) {
    // Work
  }
}
```

#### <span class="correct">Correct</span>

```okl
@kernel void myKernel(const int N) {
  const int N2 = 2 * N;
  for (int i = 0; i < N2; ++i; @outer) {
    // Work
  }
}
```

---

#### <span class="incorrect">Incorrect</span>

```okl
@kernel void myKernel(const int N) {
  int foo;
  for (int i = 0; i < N; ++i; @outer) {
    foo = i; // [Error] Can't write back to host memory
  }
}
```

#### <span class="correct">Correct</span>

```okl
@kernel void myKernel(const int N) {
  int foo = 2;
  for (int i = 0; i < N; ++i; @outer) {
    const int foo_i = foo * i;
  }
}
```
