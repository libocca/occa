
<h1 id="free">
 <a href="#/api/kernel/free" class="anchor">
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
      <a href="https://github.com/libocca/occa/blob/6d155d0c/include/occa/core/kernel.hpp#L308" target="_blank">Source</a>
    </div>
    
  </div>

  <hr>
</div>


<h2 id="description">
 <a href="#/api/kernel/free?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

Free the kernel object.
Calling [isInitialized](/api/kernel/isInitialized) will return `false` now.
