
<h1 id="operator ==">
 <a href="#/api/memory/operator_equals" class="anchor">
   <span>operator ==</span>
  </a>
</h1>

<div class="signature">

<hr>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><span class="token keyword">bool</span> operator == (<span class="token keyword">const</span> <a href="#/api/memory/">occa::memory</a> &amp;other)</code>
      <code class="mobile-only"><span class="token keyword">bool</span> operator == (
    <span class="token keyword">const</span> <a href="#/api/memory/">occa::memory</a> &amp;other
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/7d02eac1/include/occa/core/memory.hpp#L243" target="_blank">Source</a>
    </div>
    <div class="description">

      <div class="section-header">Returns</div>
      <ul class="section-list">
        <li> ::: markdown If the references are the same, this returns `true` otherwise `false`. ::: </li>
      </ul>
</div>
  </div>

<hr>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><span class="token keyword">bool</span> operator != (<span class="token keyword">const</span> <a href="#/api/memory/">occa::memory</a> &amp;other)</code>
      <code class="mobile-only"><span class="token keyword">bool</span> operator != (
    <span class="token keyword">const</span> <a href="#/api/memory/">occa::memory</a> &amp;other
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/7d02eac1/include/occa/core/memory.hpp#L256" target="_blank">Source</a>
    </div>
    <div class="description">

      <div class="section-header">Returns</div>
      <ul class="section-list">
        <li> ::: markdown If the references are different, this returns `true` otherwise `false`. ::: </li>
      </ul>
</div>
  </div>

  <hr>
</div>


<h2 id="description">
 <a href="#/api/memory/operator_equals?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

Compare if two memory objects have the same references.
