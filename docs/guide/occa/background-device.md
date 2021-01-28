# Background Device

It can be annoying to pass the `occa::device` object throughout an application, so we offer the use of a _background device_.

The _background device_ is a global `occa::device` object obtainable through:

::: tabs language

- C++

    ```cpp
    occa::device backgroundDevice = occa::getDevice();
    ```

- C

    ```c
    occaDevice backgroundDevice = occaGetDevice();
    ```

- Python

    ```python
    device = occa.get_device()
    ```

:::

?> However, the beauty of the _background device_ is not having to fetch it to make `occa::device` calls!

# Example Methods

For example, this is how we would compile kernels using the _background device_

::: tabs language

- C++

    ```cpp
    occa::kernel addVectors = occa::buildKernel("addVectors.okl",
                                                "addVectors");
    ```

- C

    ```c
    occaKernel addVectors = occaBuildKernel("addVectors.okl",
                                            "addVectors");
    ```

- Python

    ```python
    add_vectors_source = r'''
    @kernel void addVectors(const int entries,
                            const float *a,
                            const float *b,
                            float *ab) {
      for (int i = 0; i < entries; ++i; @tile(16, @outer, @inner)) {
        ab[i] = a[i] + b[i];
      }
    }
    '''

    add_vectors = occa.build_kernel_from_string(add_vectors_source,
                                                'addVectors')
    ```

:::

Similarly, we can allocate memory through _background device_ methods

::: tabs language

- C++

    ```cpp
    occa::memory o_a = occa::malloc(entries * sizeof(float), a);
    ```

- C

    ```c
    occaMemory o_a = occaMalloc(entries * sizeof(float),
                                a, occaDefault);
    ```

- Python

    ```python
    o_ab = device.malloc(a) # Where a is a numpy.ndarray
    # or
    o_ab = device.malloc(entries, dtype=np.float32)
    ```

:::

# Changing Devices

The default _background device_ is set to

::: tabs language

- C++

    ```cpp
    occa::device host = occa::host();
    ```

- C

    ```c
    occaDevice host = occaHost();
    ```

- Python

    ```python
    host = occa.host()
    ```

:::

which uses `Serial` mode and represents the machine's host.

You can set the _background device_ using an existing `occa::device` or `occa::properties` to create a new one

::: tabs language

- C++

    ```cpp
    occa::setDevice(device);
    occa::setDevice("mode: 'Serial'");
    ```

- C

    ```c
    occaSetDevice(device);
    occaSetDevice(occaString("mode: 'Serial'"));
    ```

- Python

    ```python
    occa.set_device(device)
    occa.set_device(mode='Serial')
    ```

:::

# Additional Uses

A powerful use of the _background device_ is the ability to ease the inclusion of OCCA in existing libraries.
Changing API is not always an easy process, specially when adding a library such as OCCA that targets the most computational intensive parts of the code.
Having the _background device_ implicitly allows a device to be used inside methods without passing it as a function argument.

# Thread Safety

Thread-local storage is used to store the _background device_ which create separate instances per thread but keep its use thread-safe.
