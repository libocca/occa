
<h1 id="cast">
 <a href="#/api/memory/cast" class="anchor">
   <span>cast</span>
  </a>
</h1>

<div class="signature">

<hr>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><a href="#/api/memory/">occa::memory</a> cast(<span class="token keyword">const</span> <a href="#/api/dtype_t/">occa::dtype&#95;t</a> &amp;dtype&#95;)</code>
      <code class="mobile-only"><a href="#/api/memory/">occa::memory</a> cast(
    <span class="token keyword">const</span> <a href="#/api/dtype_t/">occa::dtype&#95;t</a> &amp;dtype&#95;
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/7d02eac1/include/occa/core/memory.hpp#L439" target="_blank">Source</a>
    </div>
    <div class="description">

      <div class="section-header">Arguments</div>
      <ul class="section-list">
          
        <li>
          <strong>dtype_</strong>: ::: markdown What the return [occa::memory](/api/memory/)'s data type should be :::
        </li>

      </ul>
</div>
  </div>

  <hr>
</div>


<h2 id="description">
 <a href="#/api/memory/cast?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

Return a reference to the caller [occa::memory](/api/memory/) object but
with a different data type.
