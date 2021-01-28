
<h1 id="wrap-memory">
 <a href="#/api/device/wrapMemory" class="anchor">
   <span>wrapMemory</span>
  </a>
</h1>

<div class="signature">
  <hr>

  
  <div class="definition-container">
    <div class="definition">
      <code>occa::memory occa::device::wrapMemory(const T *ptr, const dim_t entries, const occa::json &props=occa::json())</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/26e3076e/include/occa/core/device.hpp#L645" target="_blank">Source</a>
    </div>
    <div class="description">

      <div>
        ::: markdown
        Uses the templated type to determine the type and bytes.
        :::
      </div>

      <div class="section-header">Returns</div>
      <ul class="section-list">
        <li>
          ::: markdown
          The wrapped [occa::memory](/api/memory/)
          :::
        </li>
      </ul>
    </div>

  </div>

  <hr>

  <div class="definition-container">
    <div class="definition">
      <code>occa::memory occa::device::wrapMemory(const void *ptr, const dim_t entries, const dtype_t &dtype, const occa::json &props=occa::json())</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/26e3076e/include/occa/core/device.hpp#L657" target="_blank">Source</a>
    </div>
    <div class="description">

      <div>
        ::: markdown
        Same but takes a [occa::dtype_t](/api/dtype_t) rather than a template parameter.
        :::
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
