
<h1 id="free">
 <a href="#/api/device/free" class="anchor">
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
      <a href="https://github.com/libocca/occa/blob/7d02eac1/include/occa/core/device.hpp#L256" target="_blank">Source</a>
    </div>
    
  </div>

  <hr>
</div>


<h2 id="description">
 <a href="#/api/device/free?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

Free the device, which will also free:
- Allocated [occa::memory](/api/memory/)
- Built [occa::kernel](/api/kernel/)
- Created [occa::stream](/api/stream/) and [occa::streamTag](/api/streamTag/)

Calling [isInitialized](/api/device/isInitialized) will return `false` now.
