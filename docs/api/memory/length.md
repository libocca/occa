
<h1 id="length">
 <a href="#/api/memory/length" class="anchor">
   <span>length</span>
  </a>
</h1>

<div class="signature">

<hr>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><span class="token keyword">udim&#95;t</span> length()</code>
      <code class="mobile-only"><span class="token keyword">udim&#95;t</span> length(
    
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/6d155d0c/include/occa/core/memory.hpp#L217" target="_blank">Source</a>
    </div>
    
  </div>

<hr>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><span class="token keyword">template</span> <<span class="token keyword">class</span> <span class="token keyword">T</span>>
<span class="token keyword">udim&#95;t</span> length()</code>
      <code class="mobile-only"><span class="token keyword">template</span> <<span class="token keyword">class</span> <span class="token keyword">T</span>>
<span class="token keyword">udim&#95;t</span> length(
    
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/6d155d0c/include/occa/core/memory.hpp#L228" target="_blank">Source</a>
    </div>
    <div class="description">

      <div>
        ::: markdown Same as above but explicitly chose the type (`T`) :::
      </div>
</div>
  </div>

  <hr>
</div>


<h2 id="description">
 <a href="#/api/memory/length?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

Get the length of the memory object, using its underlying [occa::dtype_t](/api/dtype_t/).
This [occa::dtype_t](/api/dtype_t/) can be fetched through the [dtype](/api/memory/dtype) method

If no type was given during [allocation](/api/device/malloc) or was ever set
through [casting it](/api/memory/cast), it will return the bytes just like [size](/api/memory/size).
