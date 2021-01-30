
<h1 id="has">
 <a href="#/api/json/has" class="anchor">
   <span>has</span>
  </a>
</h1>

<div class="signature">

<hr>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><span class="token keyword">bool</span> has(<span class="token keyword">const</span> <span class="token keyword">std::string</span> &amp;key)</code>
      <code class="mobile-only"><span class="token keyword">bool</span> has(
    <span class="token keyword">const</span> <span class="token keyword">std::string</span> &amp;key
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/6d155d0c/include/occa/types/json.hpp#L444" target="_blank">Source</a>
    </div>
    
  </div>

  <hr>
</div>


<h2 id="description">
 <a href="#/api/json/has?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

If it's an object, return whether it has a key `s` which handles paths.

For example, the following checks whether there is a nested `{a: {b: {c: ...}}}` structure:

```cpp
j.has("a/b/c")
```
