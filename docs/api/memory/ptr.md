
<h1 id="ptr">
 <a href="#/api/memory/ptr" class="anchor">
   <span>ptr</span>
  </a>
</h1>

<div class="signature">

<hr>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><span class="token keyword">template</span> <<span class="token keyword">class</span> <span class="token keyword">T</span>>
<span class="token keyword">T</span>&#42; ptr()</code>
      <code class="mobile-only"><span class="token keyword">template</span> <<span class="token keyword">class</span> <span class="token keyword">T</span>>
<span class="token keyword">T</span>&#42; ptr(
    
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/7d02eac1/include/occa/core/memory.hpp#L117" target="_blank">Source</a>
    </div>
    
  </div>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><span class="token keyword">template</span> <<span class="token keyword">class</span> <span class="token keyword">T</span>>
<span class="token keyword">const</span> <span class="token keyword">T</span>&#42; ptr()</code>
      <code class="mobile-only"><span class="token keyword">template</span> <<span class="token keyword">class</span> <span class="token keyword">T</span>>
<span class="token keyword">const</span> <span class="token keyword">T</span>&#42; ptr(
    
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/7d02eac1/include/occa/core/memory.hpp#L123" target="_blank">Source</a>
    </div>
    
  </div>

  <hr>
</div>


<h2 id="description">
 <a href="#/api/memory/ptr?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

Return the backend pointer

- _Serial_, _OpenMP_: Host pointer, which can be used in the host application
- _CUDA_, _HIP_: Allocated device pointer. If allocated with the `host: true` flag it will return the host pointer
- _OpenCL_: `cl_mem` pointer
- _Metal_: Metal buffer pointer
