
<h1 id="run">
 <a href="#/api/kernel/run" class="anchor">
   <span>run</span>
  </a>
</h1>

<div class="signature">

<hr>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><span class="token keyword">void</span> run()</code>
      <code class="mobile-only"><span class="token keyword">void</span> run(
    
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/7d02eac1/include/occa/core/kernel.hpp#L277" target="_blank">Source</a>
    </div>
    
  </div>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><span class="token keyword">void</span> run(<span class="token keyword">std::initializer_list<</span> <a href="#/api/kernelArg">occa::kernelArg</a> &gt;args)</code>
      <code class="mobile-only"><span class="token keyword">void</span> run(
    <span class="token keyword">std::initializer_list<</span> <a href="#/api/kernelArg">occa::kernelArg</a> &gt;args
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/7d02eac1/include/occa/core/kernel.hpp#L282" target="_blank">Source</a>
    </div>
    
  </div>

  <hr>
</div>


<h2 id="description">
 <a href="#/api/kernel/run?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

The more common way to run a [occa::kernel](/api/kernel/) is through [operator ()](/api/kernel/operator_parentheses).
However, we also offer a way to run a kernel by manually pushing arguments to it.
This is useful when building a kernel dynamically.

Manually push arguments through [pushArg](/api/kernel/pushArg), followed by calling this `run` function.

To clear the arguments, use [clearArgs](/api/kernel/clearArgs).
