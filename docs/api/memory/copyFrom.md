
<h1 id="copy-from">
 <a href="#/api/memory/copyFrom" class="anchor">
   <span>copyFrom</span>
  </a>
</h1>

<div class="signature">

<hr>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><span class="token keyword">void</span> copyFrom(<span class="token keyword">const</span> <span class="token keyword">void</span> &#42;src,
              <span class="token keyword">const</span> <span class="token keyword">dim&#95;t</span> bytes,
              <span class="token keyword">const</span> <span class="token keyword">dim&#95;t</span> offset,
              <span class="token keyword">const</span> <a href="#/api/json/">occa::json</a> &amp;props)</code>
      <code class="mobile-only"><span class="token keyword">void</span> copyFrom(
    <span class="token keyword">const</span> <span class="token keyword">void</span> &#42;src,
    <span class="token keyword">const</span> <span class="token keyword">dim&#95;t</span> bytes,
    <span class="token keyword">const</span> <span class="token keyword">dim&#95;t</span> offset,
    <span class="token keyword">const</span> <a href="#/api/json/">occa::json</a> &amp;props
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/6d155d0c/include/occa/core/memory.hpp#L345" target="_blank">Source</a>
    </div>
    
  </div>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><span class="token keyword">void</span> copyFrom(<span class="token keyword">const</span> <span class="token keyword">void</span> &#42;src,
              <span class="token keyword">const</span> <a href="#/api/json/">occa::json</a> &amp;props)</code>
      <code class="mobile-only"><span class="token keyword">void</span> copyFrom(
    <span class="token keyword">const</span> <span class="token keyword">void</span> &#42;src,
    <span class="token keyword">const</span> <a href="#/api/json/">occa::json</a> &amp;props
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/6d155d0c/include/occa/core/memory.hpp#L353" target="_blank">Source</a>
    </div>
    <div class="description">

      <div class="section-header">Arguments</div>
      <ul class="section-list">
          
        <li>
          <strong>src</strong>: ::: markdown Data source. :::
        </li>


        <li>
          <strong>bytes</strong>: ::: markdown How many bytes to copy. :::
        </li>


        <li>
          <strong>offset</strong>: ::: markdown The [occa::memory](/api/memory/) offset where data transfer will start. :::
        </li>


        <li>
          <strong>props</strong>: ::: markdown Any backend-specific properties for memory transfer.
For example, `async: true`. :::
        </li>

      </ul>
</div>
  </div>

<hr>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><span class="token keyword">void</span> copyFrom(<span class="token keyword">const</span> <a href="#/api/memory/">occa::memory</a> src,
              <span class="token keyword">const</span> <span class="token keyword">dim&#95;t</span> bytes,
              <span class="token keyword">const</span> <span class="token keyword">dim&#95;t</span> destOffset,
              <span class="token keyword">const</span> <span class="token keyword">dim&#95;t</span> srcOffset,
              <span class="token keyword">const</span> <a href="#/api/json/">occa::json</a> &amp;props)</code>
      <code class="mobile-only"><span class="token keyword">void</span> copyFrom(
    <span class="token keyword">const</span> <a href="#/api/memory/">occa::memory</a> src,
    <span class="token keyword">const</span> <span class="token keyword">dim&#95;t</span> bytes,
    <span class="token keyword">const</span> <span class="token keyword">dim&#95;t</span> destOffset,
    <span class="token keyword">const</span> <span class="token keyword">dim&#95;t</span> srcOffset,
    <span class="token keyword">const</span> <a href="#/api/json/">occa::json</a> &amp;props
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/6d155d0c/include/occa/core/memory.hpp#L371" target="_blank">Source</a>
    </div>
    
  </div>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><span class="token keyword">void</span> copyFrom(<span class="token keyword">const</span> <a href="#/api/memory/">occa::memory</a> src,
              <span class="token keyword">const</span> <a href="#/api/json/">occa::json</a> &amp;props)</code>
      <code class="mobile-only"><span class="token keyword">void</span> copyFrom(
    <span class="token keyword">const</span> <a href="#/api/memory/">occa::memory</a> src,
    <span class="token keyword">const</span> <a href="#/api/json/">occa::json</a> &amp;props
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/6d155d0c/include/occa/core/memory.hpp#L380" target="_blank">Source</a>
    </div>
    <div class="description">

      <div class="section-header">Arguments</div>
      <ul class="section-list">
          
        <li>
          <strong>destOffset</strong>: ::: markdown The [occa::memory](/api/memory/) offset for the caller [occa::memory](/api/memory/) :::
        </li>


        <li>
          <strong>srcOffset</strong>: ::: markdown The [occa::memory](/api/memory/) offset for the source [occa::memory](/api/memory/) (`src`) :::
        </li>

      </ul>
</div>
  </div>

  <hr>
</div>


<h2 id="description">
 <a href="#/api/memory/copyFrom?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

Copies data from the input `src` to the caller [occa::memory](/api/memory/) object
