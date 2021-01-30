
<h1 id="get-string">
 <a href="#/api/hash_t/getString" class="anchor">
   <span>getString</span>
  </a>
</h1>

<div class="signature">

<hr>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><span class="token keyword">std::string</span> getString()</code>
      <code class="mobile-only"><span class="token keyword">std::string</span> getString(
    
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/6d155d0c/include/occa/utils/hash.hpp#L133" target="_blank">Source</a>
    </div>
    
  </div>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only">operator std::string()</code>
      <code class="mobile-only">operator std::string(
    
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/6d155d0c/include/occa/utils/hash.hpp#L138" target="_blank">Source</a>
    </div>
    
  </div>

  <hr>
</div>


<h2 id="description">
 <a href="#/api/hash_t/getString?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

Return a short string representation of the hash

?> Note that this does not fully represent the hash.
?> .
?> There isn't a way to recreate the hash from just this `std::string` value
