
<h1 id="build-kernel">
 <a href="#/api/device/buildKernel" class="anchor">
   <span>buildKernel</span>
  </a>
</h1>

<div class="signature">
  <hr>

  
  <div class="definition-container">
    <div class="definition">
      <code><a href="#/api/kernel/">occa::kernel</a> buildKernel(<span class="token keyword">const</span> <span class="token keyword">std::string</span> &filename,
                         <span class="token keyword">const</span> <span class="token keyword">std::string</span> &kernelName,
                         <span class="token keyword">const</span> <a href="#/api/json/">occa::json</a> &props)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/1fea69a2/include/occa/core/device.hpp#L503" target="_blank">Source</a>
    </div>
    <div class="description">

      <div class="section-header">Returns</div>
      <ul class="section-list">
        <li>
          ::: markdown
          The compiled [occa::kernel](/api/kernel/).
          :::
        </li>
      </ul>
    </div>

  </div>


  <hr>
</div>


<h2 id="description">
 <a href="#/api/device/buildKernel?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

Builds a [occa::kernel](/api/kernel/) given a filename, kernel name, and optional properties.

<h3 id="defines">
 <a href="#/api/device/buildKernel?id=defines" class="anchor">
   <span>Defines</span>
  </a>
</h3>

Compile-time definitions can be passed through the `defines` path.
For example:

```cpp
occa::json props;
props["defines/TWO"] = 2;
```

<h3 id="includes">
 <a href="#/api/device/buildKernel?id=includes" class="anchor">
   <span>Includes</span>
  </a>
</h3>

Headers can be `#include`-ed through the `includes` path.
For example:

```cpp
occa::json props;
props["includes"].asArray();
props["includes"] += "my_header.hpp";
```

<h3 id="headers">
 <a href="#/api/device/buildKernel?id=headers" class="anchor">
   <span>Headers</span>
  </a>
</h3>

Source code can be injected through the `headers` path.
For example:

```cpp
occa::json props;
props["headers"].asArray();
props["headers"] += "#define TWO 2";
```

<h3 id="functions">
 <a href="#/api/device/buildKernel?id=functions" class="anchor">
   <span>Functions</span>
  </a>
</h3>

Lastly, [occa::function](/api/function)'s can be captured through the `functions` path.
For example:

```cpp
occa::json props;
props["functions/add"] = (
  OCCA_FUNCTION({}, [=](float a, float b) -> float {
    return a + b;
  }
);
```
