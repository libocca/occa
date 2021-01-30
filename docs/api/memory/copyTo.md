
<h1 id="copy-to">
 <a href="#/api/memory/copyTo" class="anchor">
   <span>copyTo</span>
  </a>
</h1>

<div class="signature">

<hr>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><span class="token keyword">void</span> copyTo(<span class="token keyword">void</span> &#42;dest,
            <span class="token keyword">const</span> <span class="token keyword">dim&#95;t</span> bytes,
            <span class="token keyword">const</span> <span class="token keyword">dim&#95;t</span> offset,
            <span class="token keyword">const</span> <a href="#/api/json/">occa::json</a> &amp;props)</code>
      <code class="mobile-only"><span class="token keyword">void</span> copyTo(
    <span class="token keyword">void</span> &#42;dest,
    <span class="token keyword">const</span> <span class="token keyword">dim&#95;t</span> bytes,
    <span class="token keyword">const</span> <span class="token keyword">dim&#95;t</span> offset,
    <span class="token keyword">const</span> <a href="#/api/json/">occa::json</a> &amp;props
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/6d155d0c/include/occa/core/memory.hpp#L405" target="_blank">Source</a>
    </div>
    
  </div>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><span class="token keyword">void</span> copyTo(<span class="token keyword">void</span> &#42;dest,
            <span class="token keyword">const</span> <a href="#/api/json/">occa::json</a> &amp;props)</code>
      <code class="mobile-only"><span class="token keyword">void</span> copyTo(
    <span class="token keyword">void</span> &#42;dest,
    <span class="token keyword">const</span> <a href="#/api/json/">occa::json</a> &amp;props
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/6d155d0c/include/occa/core/memory.hpp#L413" target="_blank">Source</a>
    </div>
    <div class="description">

      <div class="section-header">Arguments</div>
      <ul class="section-list">
          
        <li>
          <strong>dest</strong>: ::: markdown Where to copy the [occa::memory](/api/memory/) data to. :::
        </li>


        <li>
          <strong>bytes</strong>: ::: markdown How many bytes to copy :::
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
      <code class="desktop-only"><span class="token keyword">void</span> copyTo(<span class="token keyword">const</span> <a href="#/api/memory/">occa::memory</a> dest,
            <span class="token keyword">const</span> <span class="token keyword">dim&#95;t</span> bytes,
            <span class="token keyword">const</span> <span class="token keyword">dim&#95;t</span> destOffset,
            <span class="token keyword">const</span> <span class="token keyword">dim&#95;t</span> srcOffset,
            <span class="token keyword">const</span> <a href="#/api/json/">occa::json</a> &amp;props)</code>
      <code class="mobile-only"><span class="token keyword">void</span> copyTo(
    <span class="token keyword">const</span> <a href="#/api/memory/">occa::memory</a> dest,
    <span class="token keyword">const</span> <span class="token keyword">dim&#95;t</span> bytes,
    <span class="token keyword">const</span> <span class="token keyword">dim&#95;t</span> destOffset,
    <span class="token keyword">const</span> <span class="token keyword">dim&#95;t</span> srcOffset,
    <span class="token keyword">const</span> <a href="#/api/json/">occa::json</a> &amp;props
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/6d155d0c/include/occa/core/memory.hpp#L431" target="_blank">Source</a>
    </div>
    
  </div>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><span class="token keyword">void</span> copyTo(<span class="token keyword">const</span> <a href="#/api/memory/">occa::memory</a> dest,
            <span class="token keyword">const</span> <a href="#/api/json/">occa::json</a> &amp;props)</code>
      <code class="mobile-only"><span class="token keyword">void</span> copyTo(
    <span class="token keyword">const</span> <a href="#/api/memory/">occa::memory</a> dest,
    <span class="token keyword">const</span> <a href="#/api/json/">occa::json</a> &amp;props
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/6d155d0c/include/occa/core/memory.hpp#L440" target="_blank">Source</a>
    </div>
    <div class="description">

      <div class="section-header">Arguments</div>
      <ul class="section-list">
          
        <li>
          <strong>destOffset</strong>: ::: markdown The [occa::memory](/api/memory/) offset for the destination [occa::memory](/api/memory/) (`dest`) :::
        </li>


        <li>
          <strong>srcOffset</strong>: ::: markdown The [occa::memory](/api/memory/) offset for the caller [occa::memory](/api/memory/) :::
        </li>

      </ul>
</div>
  </div>

  <hr>
</div>


<h2 id="description">
 <a href="#/api/memory/copyTo?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

Copies data from the input `src` to the caller [occa::memory](/api/memory/) object
