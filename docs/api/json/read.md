
<h1 id="read">
 <a href="#/api/json/read" class="anchor">
   <span>read</span>
  </a>
</h1>

<div class="signature">

<hr>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><a href="#/api/json/">occa::json</a> read(<span class="token keyword">const</span> <span class="token keyword">std::string</span> &amp;filename)</code>
      <code class="mobile-only"><a href="#/api/json/">occa::json</a> read(
    <span class="token keyword">const</span> <span class="token keyword">std::string</span> &amp;filename
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/6d155d0c/include/occa/types/json.hpp#L401" target="_blank">Source</a>
    </div>
    
  </div>

  <hr>
</div>


<h2 id="description">
 <a href="#/api/json/read?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

Read the input file and parse the JSON-formatted contents.
Throw an `occa::exception` if the input is not of JSON-format

Same as [parse](/api/json/parse) but with a file rather than a `string`.
