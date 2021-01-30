
<h1 id="can-be-casted-to">
 <a href="#/api/dtype_t/canBeCastedTo" class="anchor">
   <span>canBeCastedTo</span>
  </a>
</h1>

<div class="signature">

<hr>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><span class="token keyword">bool</span> canBeCastedTo(<span class="token keyword">const</span> <a href="#/api/dtype_t/">occa::dtype&#95;t</a> &amp;other)</code>
      <code class="mobile-only"><span class="token keyword">bool</span> canBeCastedTo(
    <span class="token keyword">const</span> <a href="#/api/dtype_t/">occa::dtype&#95;t</a> &amp;other
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/6d155d0c/include/occa/dtype/dtype.hpp#L236" target="_blank">Source</a>
    </div>
    
  </div>

  <hr>
</div>


<h2 id="description">
 <a href="#/api/dtype_t/canBeCastedTo?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

Check whether flattened, two types can be matched.
For example:

- `int` can be casted to `int2` and vice-versa.
- A struct of `[int, float, int, float]` fields can be casted to a struct of `[int, float]` fields.
