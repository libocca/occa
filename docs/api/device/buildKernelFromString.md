
<h1 id="build-kernel-from-string">
 <a href="#/api/device/buildKernelFromString" class="anchor">
   <span>buildKernelFromString</span>
  </a>
</h1>

<div class="signature">

<hr>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><a href="#/api/kernel/">occa::kernel</a> buildKernelFromString(<span class="token keyword">const</span> <span class="token keyword">std::string</span> &amp;content,
                                   <span class="token keyword">const</span> <span class="token keyword">std::string</span> &amp;kernelName,
                                   <span class="token keyword">const</span> <a href="#/api/json/">occa::json</a> &amp;props)</code>
      <code class="mobile-only"><a href="#/api/kernel/">occa::kernel</a> buildKernelFromString(
    <span class="token keyword">const</span> <span class="token keyword">std::string</span> &amp;content,
    <span class="token keyword">const</span> <span class="token keyword">std::string</span> &amp;kernelName,
    <span class="token keyword">const</span> <a href="#/api/json/">occa::json</a> &amp;props
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/6d155d0c/include/occa/core/device.hpp#L537" target="_blank">Source</a>
    </div>
    <div class="description">

      <div class="section-header">Arguments</div>
      <ul class="section-list">
          
        <li>
          <strong>content</strong>: ::: markdown Source code to complile
kernelName
Specify the `@kernel` function name to use :::
        </li>


        <li>
          <strong>props</strong>: ::: markdown Backend-specific [properties](/api/json/) on how to compile the kernel.
More information in [buildKernel](/api/device/buildKernel) :::
        </li>

      </ul>

      <div class="section-header">Returns</div>
      <ul class="section-list">
        <li> ::: markdown The compiled [occa::kernel](/api/kernel/). ::: </li>
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
