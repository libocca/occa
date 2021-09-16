
<h1 id="is-struct">
 <a href="#/api/dtype_t/isStruct" class="anchor">
   <span>isStruct</span>
  </a>
</h1>

<div class="signature">

<hr>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><span class="token keyword">bool</span> isStruct()</code>
      <code class="mobile-only"><span class="token keyword">bool</span> isStruct(
    
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/7d02eac1/include/occa/dtype/dtype.hpp#L134" target="_blank">Source</a>
    </div>
    
  </div>

  <hr>
</div>


<h2 id="description">
 <a href="#/api/dtype_t/isStruct?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

Returns `true` if the data type represents a struct.
It's different that a tuple since it can keep distinct data types in its fields.
