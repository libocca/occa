# Unified Memory

Unified memory is another feature that can facilitate adding OCCA in existing codes.
Rather than working with `occa::memory` objects, we allow for the use of raw pointers instead.

::: tabs language

- C++

    ```cpp
    int *a = (int*) occa::umalloc(10 * sizeof(int));
    ```

- C

    ```c
    int *a = (int*) occaUmalloc(10 * sizeof(int));
    ```

:::

The resulting pointer indirectly maps to memory in the device.
We can edit the host pointer and use it directly in `occa::kernels`

::: tabs language

- C++

    ```cpp
    for (int i = 0; i < 10; ++i) {
      a[i] = i;
    }
    addVectors(10, a, o_b, o_ab);
    ```

- C

    ```c
    for (int i = 0; i < 10; ++i) {
      a[i] = i;
    }
    occaKernelRun(addVectors,
                  occaInt(entries),
                  a, o_b, o_ab);
    ```

:::

# Syncing Data

!> TODO: Missing Section
