
<h1 id="mode">
 <a href="#/api/memory/mode" class="anchor">
   <span>mode</span>
  </a>
</h1>

<div class="signature">

<hr>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><span class="token keyword">const</span> <span class="token keyword">std::string</span>&amp; mode()</code>
      <code class="mobile-only"><span class="token keyword">const</span> <span class="token keyword">std::string</span>&amp; mode(
    
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/7d02eac1/include/occa/core/memory.hpp#L165" target="_blank">Source</a>
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
 <a href="#/api/memory/mode?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

Returns the mode of the [occa::device](/api/device/) used to build the [occa::memory](/api/memory/).
