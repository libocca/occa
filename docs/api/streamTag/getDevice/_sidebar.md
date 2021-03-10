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
  - [occa::streamTag](/api/streamTag/)
    - [free](/api/streamTag/free)
    - [getDevice](/api/streamTag/getDevice)
    - [isInitialized](/api/streamTag/isInitialized)
    - [operator ==](/api/streamTag/operator_equals)
    - [wait](/api/streamTag/wait)
