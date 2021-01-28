
<h1 id="(constructor)">
 <a href="#/api/device/constructor" class="anchor">
   <span>(constructor)</span>
  </a>
</h1>

<div class="signature">
  <hr>

  
  <div class="definition-container">
    <div class="definition">
      <code>occa::device::device()</code>
      <div class="flex-spacing"></div>
      <a href="hi" target="_blank">Source</a>
    </div>
    <div class="description">

      <div>
        ::: markdown
        Default constructor
        :::
      </div>

  </div>

  <hr>

  <div class="definition-container">
    <div class="definition">
      <code>occa::device::device(const std::string &props)</code>
      <div class="flex-spacing"></div>
      <a href="hi" target="_blank">Source</a>
    </div>
    <div class="description">

      <div>
        ::: markdown
        Takes a JSON-formatted string for the device props.
        :::
      </div>

  </div>

  <hr>

  <div class="definition-container">
    <div class="definition">
      <code>occa::device::device(const occa::json &props)</code>
      <div class="flex-spacing"></div>
      <a href="hi" target="_blank">Source</a>
    </div>
    <div class="description">

      <div>
        ::: markdown
        Takes an [occa::json](/api/json/) argument for the device props.
        :::
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
