
<h1 id="wait">
 <a href="#/api/streamTag/wait" class="anchor">
   <span>wait</span>
  </a>
</h1>

<div class="signature">

<hr>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><span class="token keyword">void</span> wait()</code>
      <code class="mobile-only"><span class="token keyword">void</span> wait(
    
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/7d02eac1/include/occa/core/streamTag.hpp#L87" target="_blank">Source</a>
    </div>
    
  </div>

  <hr>
</div>


<h2 id="description">
 <a href="#/api/streamTag/wait?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

Wait for all queued operations on the [occa::stream](/api/stream/) before this [occa::streamTag](/api/streamTag/) was tagged.
This includes [occa::kernel](/api/kernel/) calls and [occa::memory](/api/memory/) data transfers.
