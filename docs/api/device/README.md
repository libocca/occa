<h1 id="occa::device">
 <a href="#/api/device/" class="anchor">
   <span>occa::device</span>
  </a>
</h1>

<h2 id="description">
 <a href="#/api/device/?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

A [occa::device](/api/device/) object maps to a physical device we want to program on.
Examples include a CPU, GPU, or other physical accelerator like an FPGA.

There are 2 main uses of a device:
- Memory allocation ([occa::memory](/api/memory/))
- Compile and run code ([occa::kernel](/api/kernel/))

<h2 id="setup">
 <a href="#/api/device/?id=setup" class="anchor">
   <span>Setup</span>
  </a>
</h2>

Setting up [occa::device](/api/device/) objects is done through JSON properties.
Here's an example of a CUDA device picking device `0` through its [(constructor)](/api/device/constructor):

```cpp
occa::device device({
  {"mode", "CUDA"},
  {"device_id", 0}
})
```

JSON formatted strings can also be passed directly, which can be useful when loading from a config file.

```cpp
occa::device device(
  "{ mode: 'CUDA', device_id: 0 }"
);
```

We can achieve the same using the [setup](/api/device/setup) method which take similar arguments as the constructors.
For example:

```cpp
occa::device device;
// ...
device.setup({
  {"mode", "CUDA"},
  {"device_id", 0}
})
```

<h2 id="memory allocation">
 <a href="#/api/device/?id=memory allocation" class="anchor">
   <span>Memory allocation</span>
  </a>
</h2>

We suggest allocating through the templated [malloc](/api/device/malloc) method which will keep type information around.
Here's an example which allocates memory on the device to fit a `float` array of size 10:

```cpp
occa::memory mem = device.malloc<float>(10);
```

<h2 id="kernel compilation">
 <a href="#/api/device/?id=kernel compilation" class="anchor">
   <span>Kernel compilation</span>
  </a>
</h2>

Kernel allocation can be done two ways, through [a file](/api/device/buildKernel) or [string source](/api/device/buildKernelFromString).
Here's an example which builds a [occa::kernel](/api/kernel/) from a file:

```cpp
occa::kernel addVectors = (
  device.buildKernel("addVectors.okl",
                     "addVectors")
);
```

<h2 id="interoperability">
 <a href="#/api/device/?id=interoperability" class="anchor">
   <span>Interoperability</span>
  </a>
</h2>

Lastly, we allow for interoperability with supported backends/libraries by wrapping and unwrapping memory objects.

Here's an example which takes a native pointer and wraps it as a [occa::memory](/api/memory/) object through the [wrapMemory](/api/device/wrapMemory) method:

```cpp
occa::memory occaPtr = (
  device.wrapMemory<float>(ptr, 10)
);
```

<h2 id="garbage collection">
 <a href="#/api/device/?id=garbage collection" class="anchor">
   <span>Garbage collection</span>
  </a>
</h2>

The [free](/api/device/free) function can be called to free the device along with any other object allocated by it, such as [occa::memory](/api/memory/) and [occa::kernel](/api/kernel/) objects.
OCCA implemented reference counting by default so calling [free](/api/device/free) is not required.