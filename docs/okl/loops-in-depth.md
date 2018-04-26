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

**Correct**

```okl
for (...; @outer) {
  for (int i = 0; i < 10; ++i; @inner) {
  }
  for (int i = 10; i < 20; ++i; @inner) {
  }
}
```

**Incorrect**

```okl
for (...; @outer) {
  for (int i = 0; i < 10; ++i; @inner) {
  }
  for (int i = 0; i < 20; ++i; @inner) {
  }
}
```

#### Future Plans
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

<template>
  <div class="transform-arrow">
    <v-icon>arrow_downward</v-icon>
  </div>
</template>

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

## Variable Declarations
