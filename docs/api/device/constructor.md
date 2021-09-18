
<h1 id="(constructor)">
 <a href="#/api/device/constructor" class="anchor">
   <span>(constructor)</span>
  </a>
</h1>

<div class="signature">

<hr>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only">device()</code>
      <code class="mobile-only">device(
    
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/7d02eac1/include/occa/core/device.hpp#L130" target="_blank">Source</a>
    </div>
    <div class="description">

      <div>
        ::: markdown Default constructor :::
      </div>
</div>
  </div>

<hr>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only">device(<span class="token keyword">const</span> <span class="token keyword">std::string</span> &amp;props)</code>
      <code class="mobile-only">device(
    <span class="token keyword">const</span> <span class="token keyword">std::string</span> &amp;props
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/7d02eac1/include/occa/core/device.hpp#L144" target="_blank">Source</a>
    </div>
    <div class="description">

      <div>
        ::: markdown Takes a JSON-formatted string for the device props. :::
      </div>

      <div class="section-header">Arguments</div>
      <ul class="section-list">
          
        <li>
          <strong>props</strong>: ::: markdown JSON-formatted string that defines the device properties :::
        </li>

      </ul>
</div>
  </div>

<hr>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only">device(<span class="token keyword">const</span> <a href="#/api/json/">occa::json</a> &amp;props)</code>
      <code class="mobile-only">device(
    <span class="token keyword">const</span> <a href="#/api/json/">occa::json</a> &amp;props
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/7d02eac1/include/occa/core/device.hpp#L158" target="_blank">Source</a>
    </div>
    <div class="description">

      <div>
        ::: markdown Takes an [occa::json](/api/json/) argument for the device props. :::
      </div>

      <div class="section-header">Arguments</div>
      <ul class="section-list">
          
        <li>
          <strong>props</strong>: ::: markdown Device properties :::
        </li>

      </ul>
</div>
  </div>

  <hr>
</div>


<h2 id="description">
 <a href="#/api/device/constructor?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

Creates a handle to a physical device we want to program on, such as a CPU, GPU, or other accelerator.
