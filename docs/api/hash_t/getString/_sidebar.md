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
    - [fromString](/api/hash_t/fromString)
    - [getFullString](/api/hash_t/getFullString)
    - [getInt](/api/hash_t/getInt)
    - [getString](/api/hash_t/getString)
  - [occa::json](/api/json/)
  - [occa::kernel](/api/kernel/)
  - [occa::kernelArg](/api/kernelArg)
  - [occa::memory](/api/memory/)
  - [occa::stream](/api/stream/)
  - [occa::streamTag](/api/streamTag/)
    - [isInitialized](/api/hash_t/isInitialized)
    - [operator ==](/api/hash_t/operator_equals)
    - [operator &lt;](/api/hash_t/operator_less_than)
    - [operator ^](/api/hash_t/operator_xor)
