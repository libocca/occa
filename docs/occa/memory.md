# Memory

Now that we have a device, we need to allocate and initialize memory in the device.
We cannot modify device memory directly in the host, hence data is usually initialized either by:

- Copying host data to the device
- Modifying device data inside a kernel running on the device

For our case, we'll initialize data in the host and copy it to the device.

::: tabs language

- C++

    ```cpp
    const int entries = 5;
    float *a = new float[entries];
    float *b = new float[entries];
    float *ab = new float[entries]

    for (int i = 0; i < entries; ++i) {
      a[i]  = i;
      b[i]  = 1 - i;
      ab[i] = 0;
    }
    ```

- C

    ```c
    const int entries = 5;
    float *a  = (float*) malloc(entries * sizeof(float));
    float *b  = (float*) malloc(entries * sizeof(float));
    float *ab = (float*) malloc(entries * sizeof(float));

    for (int i = 0; i < entries; ++i) {
      a[i]  = i;
      b[i]  = 1 - i;
      ab[i] = 0;
    }
    ```

:::

We now allocate and initialize device memory using our host arrays.

We initialize both `o_a` and `o_b` by copying over data from `a` and `b` respectively.
Since we'll be computing `o_ab` by summing `o_a` and `o_b`, there is no need to transfer `ab` to the device.

::: tabs language

- C++

    ```cpp
    occa::memory o_a  = device.malloc(entries * sizeof(float), a);
    occa::memory o_b  = device.malloc(entries * sizeof(float), b);
    occa::memory o_ab = device.malloc(entries * sizeof(float));
    ```

- C

    ```c
    occaMemory o_a  = occaDeviceMalloc(device, entries * sizeof(float),
                                       a, occaDefault);
    occaMemory o_b  = occaDeviceMalloc(device, entries * sizeof(float),
                                       b, occaDefault);
    occaMemory o_ab = occaDeviceMalloc(device, entries * sizeof(float),
                                       NULL, occaDefault);
    ```

:::
