
<h1 id="set">
 <a href="#/api/json/size" class="anchor">
   <span>set</span>
  </a>
</h1>

<div class="signature">

<hr>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><span class="token keyword">template</span> <<span class="token keyword">class</span> <span class="token keyword">T</span>>
<a href="#/api/json/">occa::json</a>&amp; set(<span class="token keyword">const</span> <span class="token keyword">char</span> &#42;key,
                <span class="token keyword">const</span> <span class="token keyword">T</span> &amp;value)</code>
      <code class="mobile-only"><span class="token keyword">template</span> <<span class="token keyword">class</span> <span class="token keyword">T</span>>
<a href="#/api/json/">occa::json</a>&amp; set(
    <span class="token keyword">const</span> <span class="token keyword">char</span> &#42;key,
    <span class="token keyword">const</span> <span class="token keyword">T</span> &amp;value
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/7d02eac1/include/occa/types/json.hpp#L734" target="_blank">Source</a>
    </div>
    
  </div>

  <hr>
</div>


<h2 id="description">
 <a href="#/api/json/size?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

   If it's an object, return how many key/value pairs it has.

   If it's an array, return how many array entries it has.

   If it's a string, return its length.

   Otherwise, return 0
 */
int size() const;

/**
 \xmlonly <occa-doc id="set[0]"><![CDATA[

 Description:
   Similar to [operator []](/api/json/operator_brackets) which can be used to set a value, but avoids parsing `/` as paths
