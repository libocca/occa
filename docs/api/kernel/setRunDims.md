
<h1 id="set-run-dims">
 <a href="#/api/kernel/setRunDims" class="anchor">
   <span>setRunDims</span>
  </a>
</h1>

<div class="signature">

<hr>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><span class="token keyword">void</span> setRunDims(<span class="token keyword">dim</span> outerDims,
                <span class="token keyword">dim</span> innerDims)</code>
      <code class="mobile-only"><span class="token keyword">void</span> setRunDims(
    <span class="token keyword">dim</span> outerDims,
    <span class="token keyword">dim</span> innerDims
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/7d02eac1/include/occa/core/kernel.hpp#L237" target="_blank">Source</a>
    </div>
    
  </div>

  <hr>
</div>


<h2 id="description">
 <a href="#/api/kernel/setRunDims?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

If the [occa::kernel](/api/kernel/) was compiled without OKL, the outer and inner dimensions
need to be manually set.
The dimensions are required when running modes such as `CUDA`, `HIP`, and `OpenCL`.
