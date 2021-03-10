
<h1 id="operator ^">
 <a href="#/api/hash_t/operator_xor" class="anchor">
   <span>operator ^</span>
  </a>
</h1>

<div class="signature">

<hr>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><span class="token keyword">template</span> <<span class="token keyword">class</span> <span class="token keyword">T</span>>
<a href="#/api/hash_t/">occa::hash&#95;t</a> operator ^ (<span class="token keyword">const</span> <span class="token keyword">T</span> &amp;t)</code>
      <code class="mobile-only"><span class="token keyword">template</span> <<span class="token keyword">class</span> <span class="token keyword">T</span>>
<a href="#/api/hash_t/">occa::hash&#95;t</a> operator ^ (
    <span class="token keyword">const</span> <span class="token keyword">T</span> &amp;t
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/6d155d0c/include/occa/utils/hash.hpp#L92" target="_blank">Source</a>
    </div>
    <div class="description">

      <div class="section-header">Returns</div>
      <ul class="section-list">
        <li> ::: markdown A new hash ::: </li>
      </ul>
</div>
  </div>

<hr>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><a href="#/api/hash_t/">occa::hash&#95;t</a>&amp; operator ^= (<span class="token keyword">const</span> <a href="#/api/hash_t/">occa::hash&#95;t</a> hash)</code>
      <code class="mobile-only"><a href="#/api/hash_t/">occa::hash&#95;t</a>&amp; operator ^= (
    <span class="token keyword">const</span> <a href="#/api/hash_t/">occa::hash&#95;t</a> hash
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/6d155d0c/include/occa/utils/hash.hpp#L105" target="_blank">Source</a>
    </div>
    <div class="description">

      <div class="section-header">Returns</div>
      <ul class="section-list">
        <li> ::: markdown The same [occa::hash_t](/api/hash_t/) as the caller ::: </li>
      </ul>
</div>
  </div>

  <hr>
</div>


<h2 id="description">
 <a href="#/api/hash_t/operator_xor?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

Apply a XOR (`^`) operation between two hashes, a common way to "combine" hashes
