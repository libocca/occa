
<h1 id="dont-use-refs">
 <a href="#/api/kernel/dontUseRefs" class="anchor">
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
      <a href="https://github.com/libocca/occa/blob/7d02eac1/include/occa/core/kernel.hpp#L87" target="_blank">Source</a>
    </div>
    
  </div>

  <hr>
</div>


<h2 id="description">
 <a href="#/api/kernel/dontUseRefs?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

By default, a [occa::kernel](/api/kernel/) will automatically call [free](/api/kernel/free) through reference counting.
Turn off automatic garbage collection through this method.
