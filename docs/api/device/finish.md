
<h1 id="finish">
 <a href="#/api/device/finish" class="anchor">
   <span>finish</span>
  </a>
</h1>

<div class="signature">

<hr>

  <div class="definition-container">
    <div class="definition">
      <code><span class="token keyword">void</span> finish()</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/6aadf694/include/occa/core/device.hpp#L341" target="_blank">Source</a>
    </div>
    
  </div>

  <hr>
</div>


<h2 id="description">
 <a href="#/api/device/finish?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

Finishes any asynchronous operation queued up on the device, such as
[async memory allocations](/api/device/malloc) or [kernel calls](/api/kernel/operator_parentheses).
