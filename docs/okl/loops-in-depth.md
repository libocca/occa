# Loops in Depth

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



## Variable Declarations