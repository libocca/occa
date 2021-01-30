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
    - [binaryFilename](/api/kernel/binaryFilename)
    - [clearArgs](/api/kernel/clearArgs)
    - [dontUseRefs](/api/kernel/dontUseRefs)
  - [occa::kernelArg](/api/kernelArg)
  - [occa::memory](/api/memory/)
  - [occa::stream](/api/stream/)
  - [occa::streamTag](/api/streamTag/)
    - [free](/api/kernel/free)
    - [getDevice](/api/kernel/getDevice)
    - [hash](/api/kernel/hash)
    - [isInitialized](/api/kernel/isInitialized)
    - [mode](/api/kernel/mode)
    - [name](/api/kernel/name)
    - [operator ==](/api/kernel/operator_equals)
    - [operator ()](/api/kernel/operator_parentheses)
    - [properties](/api/kernel/properties)
    - [pushArg](/api/kernel/pushArg)
    - [run](/api/kernel/run)
    - [setRunDims](/api/kernel/setRunDims)
    - [sourceFilename](/api/kernel/sourceFilename)
