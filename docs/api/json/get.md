
<h1 id="get">
 <a href="#/api/json/get" class="anchor">
   <span>get</span>
  </a>
</h1>

<div class="signature">

<hr>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><span class="token keyword">template</span> <<span class="token keyword">class</span> <span class="token keyword">T</span>>
<span class="token keyword">T</span> get(<span class="token keyword">const</span> <span class="token keyword">char</span> &#42;key,
      <span class="token keyword">const</span> <span class="token keyword">T</span> &amp;default&#95;)</code>
      <code class="mobile-only"><span class="token keyword">template</span> <<span class="token keyword">class</span> <span class="token keyword">T</span>>
<span class="token keyword">T</span> get(
    <span class="token keyword">const</span> <span class="token keyword">char</span> &#42;key,
    <span class="token keyword">const</span> <span class="token keyword">T</span> &amp;default&#95;
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/6d155d0c/include/occa/types/json.hpp#L755" target="_blank">Source</a>
    </div>
    
  </div>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><span class="token keyword">template</span> <<span class="token keyword">class</span> <span class="token keyword">T</span>>
<span class="token keyword">T</span> get(<span class="token keyword">const</span> <span class="token keyword">std::string</span> &amp;key,
      <span class="token keyword">const</span> <span class="token keyword">T</span> &amp;default&#95;)</code>
      <code class="mobile-only"><span class="token keyword">template</span> <<span class="token keyword">class</span> <span class="token keyword">T</span>>
<span class="token keyword">T</span> get(
    <span class="token keyword">const</span> <span class="token keyword">std::string</span> &amp;key,
    <span class="token keyword">const</span> <span class="token keyword">T</span> &amp;default&#95;
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/6d155d0c/include/occa/types/json.hpp#L762" target="_blank">Source</a>
    </div>
    
  </div>

  <hr>
</div>


<h2 id="description">
 <a href="#/api/json/get?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

Similar to [operator []](/api/json/operator_brackets) which can be used to get a value, but avoids parsing `/` as paths
