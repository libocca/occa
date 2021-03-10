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
  - [occa::dtype_t](/api/dtype_t/)
  - [occa::function](/api/function/)
  - [occa::hash_t](/api/hash_t/)
  - [occa::json](/api/json/)
  - [occa::kernel](/api/kernel/)
  - [occa::kernelArg](/api/kernelArg)
  - [occa::memory](/api/memory/)
    - [cast](/api/memory/cast)
    - [clone](/api/memory/clone)
    - [copyFrom](/api/memory/copyFrom)
    - [copyTo](/api/memory/copyTo)
    - [dtype](/api/memory/dtype)
    - [free](/api/memory/free)
    - [getDevice](/api/memory/getDevice)
    - [isInitialized](/api/memory/isInitialized)
    - [length](/api/memory/length)
    - [mode](/api/memory/mode)
    - [operator +](/api/memory/operator_add)
    - [operator ==](/api/memory/operator_equals)
    - [operator kernelArg](/api/memory/operator_kernelArg)
  - [occa::stream](/api/stream/)
  - [occa::streamTag](/api/streamTag/)
    - [properties](/api/memory/properties)
    - [ptr](/api/memory/ptr)
    - [size](/api/memory/size)
    - [slice](/api/memory/slice)
