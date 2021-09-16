
<h1 id="has-separate-memory-space">
 <a href="#/api/device/hasSeparateMemorySpace" class="anchor">
   <span>hasSeparateMemorySpace</span>
  </a>
</h1>

<div class="signature">

<hr>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><span class="token keyword">bool</span> hasSeparateMemorySpace()</code>
      <code class="mobile-only"><span class="token keyword">bool</span> hasSeparateMemorySpace(
    
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/7d02eac1/include/occa/core/device.hpp#L359" target="_blank">Source</a>
    </div>
    <div class="description">

      <div class="section-header">Returns</div>
      <ul class="section-list">
        <li> ::: markdown Returns `true` if the memory is directly accesible through the host. ::: </li>
      </ul>
</div>
  </div>

  <hr>
</div>


<h2 id="description">
 <a href="#/api/device/hasSeparateMemorySpace?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

Checks if the device memory is in a separate memory space than the host.
If they are not in a separate space, it should be safe to access the memory directly
in the host application.
For example, accesses of the [ptr](/api/memory/ptr) return pointer.
