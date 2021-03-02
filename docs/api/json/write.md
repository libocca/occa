
<h1 id="write">
 <a href="#/api/json/write" class="anchor">
   <span>write</span>
  </a>
</h1>

<div class="signature">

<hr>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><span class="token keyword">void</span> write(<span class="token keyword">const</span> <span class="token keyword">std::string</span> &amp;filename)</code>
      <code class="mobile-only"><span class="token keyword">void</span> write(
    <span class="token keyword">const</span> <span class="token keyword">std::string</span> &amp;filename
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/6d155d0c/include/occa/types/json.hpp#L413" target="_blank">Source</a>
    </div>
    
  </div>

  <hr>
</div>


<h2 id="description">
 <a href="#/api/json/write?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

Output the JSON-formatted string representation of the data into the given file.

Same as [dump](/api/json/dump) but writes to a file rather than a `string`.
