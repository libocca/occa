<div class="api-version-container">
  <select onchange="vm.onLanguageChange(this)">
    <option value="cpp">C++</option>
  </select>
  <select onchange="vm.onVersionChange(this)">
    <option value="nightly">Nightly</option>
  </select>
</div>

- [**API**](/api/)
  - [occa::device](/api/device/)
    - [(constructor)](/api/device/constructor)
    - [buildKernel](/api/device/buildKernel)
    - [buildKernelFromString](/api/device/buildKernelFromString)
    - [createStream](/api/device/createStream)
    - [dontUseRefs](/api/device/dontUseRefs)
    - [finish](/api/device/finish)
    - [free](/api/device/free)
    - [getStream](/api/device/getStream)
    - [hasSeparateMemorySpace](/api/device/hasSeparateMemorySpace)
    - [hash](/api/device/hash)
    - [isInitialized](/api/device/isInitialized)
    - [malloc](/api/device/malloc)
    - [memoryAllocated](/api/device/memoryAllocated)
    - [memorySize](/api/device/memorySize)
    - [mode](/api/device/mode)
    - [operator ==](/api/device/operator_equals)
    - [properties](/api/device/properties)
    - [setStream](/api/device/setStream)
    - [setup](/api/device/setup)
    - [tagStream](/api/device/tagStream)
    - [timeBetween](/api/device/timeBetween)
    - [waitFor](/api/device/waitFor)
    - [wrapMemory](/api/device/wrapMemory)
  - [occa::dtype_t](/api/dtype_t/)
  - [occa::function](/api/function/)
  - [occa::hash_t](/api/hash_t/)
  - [occa::json](/api/json/)
  - [occa::kernel](/api/kernel/)
  - [occa::kernelArg](/api/kernelArg)
  - [occa::memory](/api/memory/)
  - [occa::stream](/api/stream/)
  - [occa::streamTag](/api/streamTag/)
