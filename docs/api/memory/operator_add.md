
<h1 id="operator +">
 <a href="#/api/memory/operator_add" class="anchor">
   <span>operator +</span>
  </a>
</h1>

<div class="signature">

<hr>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><a href="#/api/memory/">occa::memory</a> operator + (<span class="token keyword">const</span> <span class="token keyword">dim&#95;t</span> offset)</code>
      <code class="mobile-only"><a href="#/api/memory/">occa::memory</a> operator + (
    <span class="token keyword">const</span> <span class="token keyword">dim&#95;t</span> offset
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/7d02eac1/include/occa/core/memory.hpp#L269" target="_blank">Source</a>
    </div>
    
  </div>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><a href="#/api/memory/">occa::memory</a>&amp; operator += (<span class="token keyword">const</span> <span class="token keyword">dim&#95;t</span> offset)</code>
      <code class="mobile-only"><a href="#/api/memory/">occa::memory</a>&amp; operator += (
    <span class="token keyword">const</span> <span class="token keyword">dim&#95;t</span> offset
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/7d02eac1/include/occa/core/memory.hpp#L274" target="_blank">Source</a>
    </div>
    <div class="description">

      <div class="section-header">Returns</div>
      <ul class="section-list">
        <li> ::: markdown A [occa::memory](/api/memory/) object shifted by `offset` bytes ::: </li>
      </ul>
</div>
  </div>

  <hr>
</div>


<h2 id="description">
 <a href="#/api/memory/operator_add?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

Same as calling [slice](/api/memory/slice)`(offset)`
