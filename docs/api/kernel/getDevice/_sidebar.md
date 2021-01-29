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
  - [occa::kernelArg](/api/kernelArg)
  - [occa::kernel](/api/kernel/)
    - [binaryFilename](/api/kernel/binaryFilename)
    - [clearArgs](/api/kernel/clearArgs)
    - [dontUseRefs](/api/kernel/dontUseRefs)
    - [free](/api/kernel/free)
    - [getDevice](/api/kernel/getDevice)
  - [occa::device](/api/device/)
  - [occa::function](/api/function)
  - [occa::stream](/api/stream)
  - [occa::dtype_t](/api/dtype_t)
  - [occa::memory](/api/memory/)
  - [occa::streamTag](/api/streamTag)
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
