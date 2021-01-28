<div class="api-version-container">
  <select onchange="vm.onLanguageChange(this)">
    <option value="cpp">C++</option>
  </select>
  <select onchange="vm.onVersionChange(this)">
    <option value="nightly">Nightly</option>
  </select>
</div>

- [**API**](/api/)
  - [occa::json](/api/json/)
  - [occa::hash_t](/api/hash_t)
  - [occa::kernel](/api/kernel/)
  - [occa::device](/api/device/)
  - [occa::function](/api/function)
  - [occa::stream](/api/stream)
  - [occa::dtype_t](/api/dtype_t)
  - [occa::memory](/api/memory/)
    - [(constructor)](/api/memory/constructor)
    - [cast](/api/memory/cast)
    - [clone](/api/memory/clone)
    - [copyFrom](/api/memory/copyFrom)
    - [copyTo](/api/memory/copyTo)
  - [occa::streamTag](/api/streamTag)
    - [free](/api/memory/free)
    - [ptr](/api/memory/ptr)
    - [slice](/api/memory/slice)
