<h1 id="occa::dtype_t">
 <a href="#/api/dtype_t/" class="anchor">
   <span>occa::dtype_t</span>
  </a>
</h1>

<h2 id="description">
 <a href="#/api/dtype_t/?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

Represents a data type, such as:
- `occa::dtype::void_` &rarr; `void`
- `occa::dtype::float_` &rarr; `float`
- `occa::dtype::byte` &rarr; A wildcard type, matching anything

[occa::dtype_t](/api/dtype_t/) data types are used to hold type information on many things, such as
[occa::memory](/api/memory/) object types or [occa::kernel](/api/kernel/) argument types.