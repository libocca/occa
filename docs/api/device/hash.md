
<h1 id="hash">
 <a href="#/api/device/hash" class="anchor">
   <span>hash</span>
  </a>
</h1>

<div class="signature">

<hr>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><a href="#/api/hash_t/">occa::hash&#95;t</a> hash()</code>
      <code class="mobile-only"><a href="#/api/hash_t/">occa::hash&#95;t</a> hash(
    
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/6d155d0c/include/occa/core/device.hpp#L306" target="_blank">Source</a>
    </div>
    <div class="description">

      <div class="section-header">Returns</div>
      <ul class="section-list">
        <li> ::: markdown The device [hash](/api/hash_t/) ::: </li>
      </ul>
</div>
  </div>

  <hr>
</div>


<h2 id="description">
 <a href="#/api/device/hash?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

Gets the [hash](/api/hash_t/) of the device.
Two devices should have the same hash if they point to the same hardware device
and setup with the same properties.
