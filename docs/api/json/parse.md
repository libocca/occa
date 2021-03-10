
<h1 id="parse">
 <a href="#/api/json/parse" class="anchor">
   <span>parse</span>
  </a>
</h1>

<div class="signature">

<hr>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><a href="#/api/json/">occa::json</a> parse(<span class="token keyword">const</span> <span class="token keyword">char</span> &#42;&amp;c)</code>
      <code class="mobile-only"><a href="#/api/json/">occa::json</a> parse(
    <span class="token keyword">const</span> <span class="token keyword">char</span> &#42;&amp;c
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/6d155d0c/include/occa/types/json.hpp#L383" target="_blank">Source</a>
    </div>
    
  </div>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><a href="#/api/json/">occa::json</a> parse(<span class="token keyword">const</span> <span class="token keyword">std::string</span> &amp;s)</code>
      <code class="mobile-only"><a href="#/api/json/">occa::json</a> parse(
    <span class="token keyword">const</span> <span class="token keyword">std::string</span> &amp;s
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/6d155d0c/include/occa/types/json.hpp#L388" target="_blank">Source</a>
    </div>
    
  </div>

  <hr>
</div>


<h2 id="description">
 <a href="#/api/json/parse?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

Parse a JSON-formatted string.
Throw an `occa::exception` if the input is not of JSON-format

Same as [read](/api/json/read) but with a file rather than a `string`.
