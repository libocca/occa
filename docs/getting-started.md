# Getting Started

The OCCA source code can be found in |Github Link| under an MIT License

Below are the list of dependencies

::: language-tabs

```cpp
for (int i = 0; i < 10; ++i) {
}
```

```c
for (int i = 0; i < 10; ++i) {
}
```

:::

::: language-tabs

```cpp
for (int i = 0; i < 20; ++i) {
}
```

```c
for (int i = 0; i < 20; ++i) {
}
```

:::

::: os-tabs

- linux
```cpp
for (int i = 0; i < 10; ++i) {
}
```

- macos
```c
for (int i = 0; i < 10; ++i) {
}
```

- windows
```python
for (int i = 0; i < 10; ++i) {
}
```

:::

.. tabs::

   .. group-tab:: Linux

      - C++ compiler
      - The :code:`make` command (in the :code:`build-essential` package)

   .. group-tab:: Mac OSX

      - C++ compiler
      - The :code:`make` command (in the :code:`build-essential` package)

   .. group-tab:: Windows

      - Visual Studio


# Download and Installation

Use git to download OCCA or download the latest release :code:`.tar.gz` from |Github Release Link|

.. tabs::

   .. code-tab:: bash Linux

      git clone https://github.com/libocca/occa.git
      cd occa
      make -j4

   .. code-tab:: bash Mac OSX

      git clone https://github.com/libocca/occa.git
      cd occa
      make -j4

   .. code-tab:: bash Windows

      ???


# Testing Installation

OCCA comes with a command called :code:`occa`, found inside the :code:`bin` directory.
The purpose of :code:`occa` is to help gather device information as well as other useful utilities

For more information, please see the section on `Command-Line Interface <../cli.html>`_

.. tabs::

   .. group-tab:: Linux

      .. code-block:: bash

         export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${PWD}/lib"
         PATH+=":${PWD}/bin"
         occa info

      Output

      .. code-block:: bash

         ==========o======================o==========================================
          CPU Info | Cores                | 16
                   | Memory (RAM)         | 31 GB
                   | Clock Frequency      | 32.623 GHz
                   | SIMD Instruction Set | SSE2
                   | SIMD Width           | 128 bits
         ==========o======================o==========================================
          OpenCL   | Device Name          | GeForce GTX 980
                   | Driver Vendor        | NVIDIA
                   | Platform ID          | 0
                   | Device ID            | 0
                   | Memory               | 3 GB
                   |----------------------|------------------------------------------
                   | Device Name          | GeForce GTX 980
                   | Driver Vendor        | NVIDIA
                   | Platform ID          | 0
                   | Device ID            | 1
                   | Memory               | 3 GB
                   |----------------------|------------------------------------------
                   | Device Name          | GeForce GTX 980
                   | Driver Vendor        | NVIDIA
                   | Platform ID          | 0
                   | Device ID            | 2
                   | Memory               | 3 GB
                   |----------------------|------------------------------------------
                   | Device Name          | Intel(R) Core(TM) i7-5960X CPU @ 3.00GHz
                   | Driver Vendor        | Intel
                   | Platform ID          | 1
                   | Device ID            | 0
                   | Memory               | 31 GB
         ==========o======================o==========================================
          CUDA     | Device ID            | 0
                   | Device Name          | GeForce GTX 980
                   | Memory               | 3 GB
                   |----------------------|------------------------------------------
                   | Device ID            | 1
                   | Device Name          | GeForce GTX 980
                   | Memory               | 3 GB
                   |----------------------|------------------------------------------
                   | Device ID            | 2
                   | Device Name          | GeForce GTX 980
                   | Memory               | 3 GB
         ==========o======================o==========================================

   .. group-tab:: Mac OSX

      .. code-block:: bash

         export DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH}:${PWD}/lib"
         PATH+=":${PWD}/bin"
         occa info

      Output

      .. code-block:: bash

         ==========o======================o==========================================
          CPU Info | Processor Name       | Intel(R) Core(TM) i5-4260U CPU @ 1.40GHz
                   | Cores                | 4
                   | Memory (RAM)         | 4 GB
                   | Clock Frequency      | 1.4 GHz
                   | SIMD Instruction Set | SSE4
                   | SIMD Width           | 128 bits
                   | L1 Cache Size (d)    |  32 KB
                   | L2 Cache Size        | 256 KB
                   | L3 Cache Size        |   3 MB
         ==========o======================o==========================================
          OpenCL   | Device Name          | Intel(R) Core(TM) i5-4260U CPU @ 1.40GHz
                   | Driver Vendor        | Intel
                   | Platform ID          | 0
                   | Device ID            | 0
                   | Memory               | 4 GB
                   |----------------------|------------------------------------------
                   | Device Name          | HD Graphics 5000
                   | Driver Vendor        | Intel
                   | Platform ID          | 0
                   | Device ID            | 1
                   | Memory               | 1 GB
         ==========o======================o==========================================

# Run Example

The simplest example we have is called AddVectors where two vectors are instantiated and added in a device

.. tabs::

   .. group-tab:: Linux

      .. code-block:: bash

         cd examples/addVectors/cpp
         make
         ./main

      Output

      .. code-block:: bash

         ==========o======================o==========================================
          CPU Info | Cores                | 16
                   | Memory (RAM)         | 31 GB
                   | Clock Frequency      | 32.684 GHz
                   | SIMD Instruction Set | SSE2
                   | SIMD Width           | 128 bits
         ==========o======================o==========================================
          OpenCL   | Device Name          | GeForce GTX 980
                   | Driver Vendor        | NVIDIA
                   | Platform ID          | 0
                   | Device ID            | 0
                   | Memory               | 3 GB
                   |----------------------|------------------------------------------
                   | Device Name          | GeForce GTX 980
                   | Driver Vendor        | NVIDIA
                   | Platform ID          | 0
                   | Device ID            | 1
                   | Memory               | 3 GB
                   |----------------------|------------------------------------------
                   | Device Name          | GeForce GTX 980
                   | Driver Vendor        | NVIDIA
                   | Platform ID          | 0
                   | Device ID            | 2
                   | Memory               | 3 GB
                   |----------------------|------------------------------------------
                   | Device Name          | Intel(R) Core(TM) i7-5960X CPU @ 3.00GHz
                   | Driver Vendor        | Intel
                   | Platform ID          | 1
                   | Device ID            | 0
                   | Memory               | 31 GB
         ==========o======================o==========================================
          CUDA     | Device ID            | 0
                   | Device Name          | GeForce GTX 980
                   | Memory               | 3 GB
                   |----------------------|------------------------------------------
                   | Device ID            | 1
                   | Device Name          | GeForce GTX 980
                   | Memory               | 3 GB
                   |----------------------|------------------------------------------
                   | Device ID            | 2
                   | Device Name          | GeForce GTX 980
                   | Memory               | 3 GB
         ==========o======================o==========================================
         Compiling [addVectors]
         clang++ -x c++ -fPIC -shared -g /home/dsm5/.occa/cache/e60679bfca62c2f2/source.occa -o /home/dsm5/.occa/cache/e60679bfca62c2f2/binary -I/home/dsm5/git/night/include -L/home/dsm5/git/night/lib -locca

         Compiling [addVectors0]
         clang++ -x c++ -fPIC -shared -g /home/dsm5/.occa/cache/94976c92c3964442/source.occa -o /home/dsm5/.occa/cache/94976c92c3964442/binary -I/home/dsm5/git/night/include -L/home/dsm5/git/night/lib -locca

         0: 1
         1: 1
         2: 1
         3: 1
         4: 1

   .. group-tab:: Mac OSX

      .. code-block:: bash

         cd examples/addVectors/cpp
         make
         ./main

      Output

      .. code-block:: bash

         ==========o======================o==========================================
          CPU Info | Processor Name       | Intel(R) Core(TM) i5-4260U CPU @ 1.40GHz
                   | Cores                | 4
                   | Memory (RAM)         | 4 GB
                   | Clock Frequency      | 1.4 GHz
                   | SIMD Instruction Set | SSE4
                   | SIMD Width           | 128 bits
                   | L1 Cache Size (d)    |  32 KB
                   | L2 Cache Size        | 256 KB
                   | L3 Cache Size        |   3 MB
         ==========o======================o==========================================
          OpenCL   | Device Name          | Intel(R) Core(TM) i5-4260U CPU @ 1.40GHz
                   | Driver Vendor        | Intel
                   | Platform ID          | 0
                   | Device ID            | 0
                   | Memory               | 4 GB
                   |----------------------|------------------------------------------
                   | Device Name          | HD Graphics 5000
                   | Driver Vendor        | Intel
                   | Platform ID          | 0
                   | Device ID            | 1
                   | Memory               | 1 GB
         ==========o======================o==========================================
         Compiling [addVectors]
         clang++ -x c++ -fPIC -shared -I. -D__extern_always_inline=inline -O3 -mtune=native -ftree-vectorize -funroll-loops -ffast-math /Users/dsm5/.occa/cache/4c38ebbf648a4b23/source.occa -o /Users/dsm5/.occa/cache/4c38ebbf648a4b23/binary -I/Users/dsm5/git/night/include -L/Users/dsm5/git/night/lib -locca

         Compiling [addVectors0]
         clang++ -x c++ -fPIC -shared -I. -D__extern_always_inline=inline -O3 -mtune=native -ftree-vectorize -funroll-loops -ffast-math /Users/dsm5/.occa/cache/3ea9fe9264150348/source.occa -o /Users/dsm5/.occa/cache/3ea9fe9264150348/binary -I/Users/dsm5/git/night/include -L/Users/dsm5/git/night/lib -locca

         0: 1
         1: 1
         2: 1
         3: 1
         4: 1

   .. code-tab:: bash Windows

      ???

.. note::

   Compiled kernels are cached for fast consecutive builds

Note that when running addVectors is run a second time, the compilation info changes.
Compiled kernels are cached and its binaries are reused if nothing changed in the compilation step (e.g. device information, kernel defines, etc)


.. tabs::

   .. group-tab:: Linux

      First run

      .. code-block:: bash

         Compiling [addVectors]
         clang++ -x c++ -fPIC -shared -g /home/dsm5/.occa/cache/e60679bfca62c2f2/source.occa -o /home/dsm5/.occa/cache/e60679bfca62c2f2/binary -I/home/dsm5/git/night/include -L/home/dsm5/git/night/lib -locca

         Compiling [addVectors0]
         clang++ -x c++ -fPIC -shared -g /home/dsm5/.occa/cache/94976c92c3964442/source.occa -o /home/dsm5/.occa/cache/94976c92c3964442/binary -I/home/dsm5/git/night/include -L/home/dsm5/git/night/lib -locca

      Second run

      .. code-block:: bash

         Found cached binary of [2d38aae833d7e36a/parsedSource.occa] in [e60679bfca62c2f2/binary]
         Found cached binary of [2d38aae833d7e36a/parsedSource.occa] in [94976c92c3964442/binary]

   .. group-tab:: Mac OSX

      First run

      .. code-block:: bash

         Compiling [addVectors]
         clang++ -x c++ -fPIC -shared -I. -D__extern_always_inline=inline -O3 -mtune=native -ftree-vectorize -funroll-loops -ffast-math /Users/dsm5/.occa/cache/4c38ebbf648a4b23/source.occa -o /Users/dsm5/.occa/cache/4c38ebbf648a4b23/binary -I/Users/dsm5/git/night/include -L/Users/dsm5/git/night/lib -locca

         Compiling [addVectors0]
         clang++ -x c++ -fPIC -shared -I. -D__extern_always_inline=inline -O3 -mtune=native -ftree-vectorize -funroll-loops -ffast-math /Users/dsm5/.occa/cache/3ea9fe9264150348/source.occa -o /Users/dsm5/.occa/cache/3ea9fe9264150348/binary -I/Users/dsm5/git/night/include -L/Users/dsm5/git/night/lib -locca

      Second run

      .. code-block:: bash

         Found cached binary of [2d38aae833d7e36a/parsedSource.occa] in [4c38ebbf648a4b23/binary]
         Found cached binary of [2d38aae833d7e36a/parsedSource.occa] in [3ea9fe9264150348/binary]

   .. code-tab:: bash Windows

      ???


.. |Github Link| raw:: html

   <a href="https://github.com/libocca/occa" target="_blank">
      <i class="github icon" style="margin-right: 0;"></i>
      Github
   </a>

.. |Github Release Link| raw:: html

   <a href="https://github.com/libocca/occa/releases" target="_blank">
      <i class="github icon" style="margin-right: 0;"></i>
      Github
   </a>
