
<h1 id="finish">
 <a href="#/api/device/finish" class="anchor">
   <span>finish</span>
  </a>
</h1>

<div class="signature">
  <hr>

  
  <div class="definition-container">
    <div class="definition">
      <code>void occa::device::finish()</code>
      <div class="flex-spacing"></div>
      <a href="hi" target="_blank">Source</a>
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
