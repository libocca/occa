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
    - [addField](/api/dtype_t/addField)
    - [bytes](/api/dtype_t/bytes)
    - [canBeCastedTo](/api/dtype_t/canBeCastedTo)
    - [isStruct](/api/dtype_t/isStruct)
    - [isTuple](/api/dtype_t/isTuple)
    - [matches](/api/dtype_t/matches)
    - [name](/api/dtype_t/name)
    - [operator []](/api/dtype_t/operator_bracket)
    - [operator ==](/api/dtype_t/operator_equals)
  - [occa::function](/api/function/)
  - [occa::hash_t](/api/hash_t/)
  - [occa::json](/api/json/)
  - [occa::kernel](/api/kernel/)
  - [occa::kernelArg](/api/kernelArg)
  - [occa::memory](/api/memory/)
  - [occa::stream](/api/stream/)
  - [occa::streamTag](/api/streamTag/)
    - [registerType](/api/dtype_t/registerType)
    - [structFieldCount](/api/dtype_t/structFieldCount)
    - [structFieldNames](/api/dtype_t/structFieldNames)
    - [tupleSize](/api/dtype_t/tupleSize)
