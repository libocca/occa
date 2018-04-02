# Device

We start off by connecting to a physical device through an OCCA device object.

::: tabs language

- C++

    ```cpp
    #include "occa.hpp"
    occa::device device("mode: 'Serial'");
    ```

- C

    ```c
    #include "occa.h"
    occaDevice device = occaCreateDevice("mode: 'Serial'");
    ```

:::

The string used to initialize the device

```cpp
"mode: 'Serial'"
```

creates an `occa::properties` object which is then passed to the device. <!-- TODO -->
Properties uses JSON format with some shorthand notations found in JavaScript.

```js
{
  mode: 'Serial'
}
```

The only property field required by OCCA when creating a device is `mode`.
However, each `mode` has its own requirements, such as CUDA requiring a `deviceID`.
In this case, we're initializing a device that runs code serially.

?> _Serial_ mode is useful for debugging buggy kernel code

Here are examples for the all core modes supported in OCCA.

::: tabs backend

- Serial

    ```cpp
    "mode: 'Serial'"
    ```

    ```js
    {
      mode: 'Serial'
    }
    ```

- OpenMP

    ```cpp
    "mode: 'OpenMP', threads: 4"
    ```

    ```js
    {
      mode: 'OpenMP',
      threads: 4
    }
    ```

- Threads

    ```cpp
    "mode: 'Serial', threads: 4, pinnedCores: [0, 1, 2, 3]"
    ```

    ```js
    {
      mode: 'Serial',
      threads: 4,
      pinnedCores: [0, 1, 2, 3]
    }
    ```

- OpenCL

    ```cpp
    "mode: 'OpenCL', deviceID: 0, platformID: 0"
    ```

    ```js
    {
      mode: 'OpenCL',
      deviceID: 0,
      platformID: 0
    }
    ```

- CUDA

    ```cpp
    "mode: 'CUDA', deviceID: 0"
    ```

    ```js
    {
      mode: 'CUDA',
      deviceID: 0
    }
    ```

:::
