
<h1 id="wrap-memory">
 <a href="#/api/device/wrapMemory" class="anchor">
   <span>wrapMemory</span>
  </a>
</h1>

<div class="signature">

<hr>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><span class="token keyword">template</span> <<span class="token keyword">class</span> <span class="token keyword">T</span>>
<a href="#/api/memory/">occa::memory</a> wrapMemory(<span class="token keyword">const</span> <span class="token keyword">T</span> &#42;ptr,
                        <span class="token keyword">const</span> <span class="token keyword">dim&#95;t</span> entries,
                        <span class="token keyword">const</span> <a href="#/api/json/">occa::json</a> &amp;props)</code>
      <code class="mobile-only"><span class="token keyword">template</span> <<span class="token keyword">class</span> <span class="token keyword">T</span>>
<a href="#/api/memory/">occa::memory</a> wrapMemory(
    <span class="token keyword">const</span> <span class="token keyword">T</span> &#42;ptr,
    <span class="token keyword">const</span> <span class="token keyword">dim&#95;t</span> entries,
    <span class="token keyword">const</span> <a href="#/api/json/">occa::json</a> &amp;props
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/6d155d0c/include/occa/core/device.hpp#L685" target="_blank">Source</a>
    </div>
    <div class="description">

      <div>
        ::: markdown Uses the templated type to determine the type and bytes. :::
      </div>

      <div class="section-header">Returns</div>
      <ul class="section-list">
        <li> ::: markdown The wrapped [occa::memory](/api/memory/) ::: </li>
      </ul>
</div>
  </div>

<hr>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><a href="#/api/memory/">occa::memory</a> wrapMemory(<span class="token keyword">const</span> <span class="token keyword">void</span> &#42;ptr,
                        <span class="token keyword">const</span> <span class="token keyword">dim&#95;t</span> entries,
                        <span class="token keyword">const</span> <a href="#/api/dtype_t/">occa::dtype&#95;t</a> &amp;dtype,
                        <span class="token keyword">const</span> <a href="#/api/json/">occa::json</a> &amp;props)</code>
      <code class="mobile-only"><a href="#/api/memory/">occa::memory</a> wrapMemory(
    <span class="token keyword">const</span> <span class="token keyword">void</span> &#42;ptr,
    <span class="token keyword">const</span> <span class="token keyword">dim&#95;t</span> entries,
    <span class="token keyword">const</span> <a href="#/api/dtype_t/">occa::dtype&#95;t</a> &amp;dtype,
    <span class="token keyword">const</span> <a href="#/api/json/">occa::json</a> &amp;props
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/6d155d0c/include/occa/core/device.hpp#L697" target="_blank">Source</a>
    </div>
    <div class="description">

      <div>
        ::: markdown Same but takes a [occa::dtype_t](/api/dtype_t/) rather than a template parameter. :::
      </div>
</div>
  </div>

  <hr>
</div>


<h2 id="description">
 <a href="#/api/device/wrapMemory?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

Wrap a native backend pointer inside a [occa::memory](/api/memory/) for the device.
The simplest example would be on a `Serial` or `OpenMP` device, where a regular pointer allocated through `malloc` or `new` is passed in.
For other modes, such as CUDA or HIP, it takes the pointer allocated through their API.

> Note that automatic garbage collection is not set for wrapped memory objects.
