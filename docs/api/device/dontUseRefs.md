
<h1 id="dont-use-refs">
 <a href="#/api/device/dontUseRefs" class="anchor">
   <span>dontUseRefs</span>
  </a>
</h1>

<div class="signature">

<hr>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><span class="token keyword">void</span> dontUseRefs()</code>
      <code class="mobile-only"><span class="token keyword">void</span> dontUseRefs(
    
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/7d02eac1/include/occa/core/device.hpp#L185" target="_blank">Source</a>
    </div>
    
  </div>

  <hr>
</div>


<h2 id="description">
 <a href="#/api/device/dontUseRefs?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

By default, a [occa::device](/api/device/) will automatically call [free](/api/device/free) through reference counting.
Turn off automatic garbage collection through this method.
