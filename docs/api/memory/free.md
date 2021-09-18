
<h1 id="free">
 <a href="#/api/memory/free" class="anchor">
   <span>free</span>
  </a>
</h1>

<div class="signature">

<hr>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><span class="token keyword">void</span> free()</code>
      <code class="mobile-only"><span class="token keyword">void</span> free(
    
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/7d02eac1/include/occa/core/memory.hpp#L460" target="_blank">Source</a>
    </div>
    
  </div>

  <hr>
</div>


<h2 id="description">
 <a href="#/api/memory/free?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

Free the device memory.
Calling [isInitialized](/api/memory/isInitialized) will return `false` now.
