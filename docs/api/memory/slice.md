
<h1 id="slice">
 <a href="#/api/memory/slice" class="anchor">
   <span>slice</span>
  </a>
</h1>

<div class="signature">

<hr>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><a href="#/api/memory/">occa::memory</a> slice(<span class="token keyword">const</span> <span class="token keyword">dim&#95;t</span> offset,
                   <span class="token keyword">const</span> <span class="token keyword">dim&#95;t</span> count)</code>
      <code class="mobile-only"><a href="#/api/memory/">occa::memory</a> slice(
    <span class="token keyword">const</span> <span class="token keyword">dim&#95;t</span> offset,
    <span class="token keyword">const</span> <span class="token keyword">dim&#95;t</span> count
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/6d155d0c/include/occa/core/memory.hpp#L320" target="_blank">Source</a>
    </div>
    
  </div>

  <hr>
</div>


<h2 id="description">
 <a href="#/api/memory/slice?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

Returns a [occa::memory](/api/memory/) object with the same reference as the caller,
but has its start and end pointer values shifted.

For example:

```cpp
// mem = {?, ?, ?, ?}
occa::memory mem = device.malloc<float>(4);

occa::memory firstHalf = mem.slice(0, 2);
occa::memory lastHalf = mem.slice(2, 4); // Or just mem.slice(2)

int values[4] = {1, 2, 3, 4}

// mem = {1, 2, ?, ?}
firstHalf.copyFrom(values);

// mem = {1, 2, 3, 4}
secondtHalf.copyFrom(values + 2);
```
