
<h1 id="mode">
 <a href="#/api/device/mode" class="anchor">
   <span>mode</span>
  </a>
</h1>

<div class="signature">
  <hr>

  
  <div class="definition-container">
    <div class="definition">
      <code><span class="token keyword">const</span> <span class="token keyword">std::string</span>& mode()</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/a7d71df6/include/occa/core/device.hpp#L267" target="_blank">Source</a>
    </div>
    <div class="description">

      <div class="section-header">Returns</div>
      <ul class="section-list">
        <li> ::: markdown The `mode` string, such as `"Serial"`, `"CUDA"`, or `"HIP"`. ::: </li>
      </ul>
</div>
  </div>


  <hr>
</div>


<h2 id="description">
 <a href="#/api/device/mode?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

Returns the device mode, matching the backend the device is targeting.