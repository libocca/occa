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
  - [occa::stream](/api/stream/)
    - [free](/api/stream/free)
    - [getDevice](/api/stream/getDevice)
    - [isInitialized](/api/stream/isInitialized)
    - [mode](/api/stream/mode)
    - [operator ==](/api/stream/operator_equals)
    - [properties](/api/stream/properties)
  - [occa::streamTag](/api/streamTag/)
