
<h1 id="build-kernel-from-string">
 <a href="#/api/device/buildKernelFromString" class="anchor">
   <span>buildKernelFromString</span>
  </a>
</h1>

<div class="signature">
  <hr>

  
  <div class="definition-container">
    <div class="definition">
      <code><a href="#/api/kernel/">occa::kernel</a> buildKernelFromString(<span class="token keyword">const</span> <span class="token keyword">std::string</span> &content,
                                   <span class="token keyword">const</span> <span class="token keyword">std::string</span> &kernelName,
                                   <span class="token keyword">const</span> <a href="#/api/json/">occa::json</a> &props)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/1202d27b/include/occa/core/device.hpp#L518" target="_blank">Source</a>
    </div>
    <div class="description">

      <div class="section-header">Returns</div>
      <ul class="section-list">
        <li>
          ::: markdown
          The compiled [occa::kernel](/api/kernel/).
          :::
        </li>
      </ul>
    </div>

  </div>


  <hr>
</div>


<h2 id="description">
 <a href="#/api/device/buildKernelFromString?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

Same as [buildKernel](/api/device/buildKernel) but given the kernel source code rather than the filename.
