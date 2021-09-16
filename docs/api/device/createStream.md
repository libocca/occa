
<h1 id="create-stream">
 <a href="#/api/device/createStream" class="anchor">
   <span>createStream</span>
  </a>
</h1>

<div class="signature">

<hr>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><a href="#/api/stream/">occa::stream</a> createStream(<span class="token keyword">const</span> <a href="#/api/json/">occa::json</a> &amp;props)</code>
      <code class="mobile-only"><a href="#/api/stream/">occa::stream</a> createStream(
    <span class="token keyword">const</span> <a href="#/api/json/">occa::json</a> &amp;props
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/7d02eac1/include/occa/core/device.hpp#L377" target="_blank">Source</a>
    </div>
    <div class="description">

      <div class="section-header">Returns</div>
      <ul class="section-list">
        <li> ::: markdown Newly created [occa::stream](/api/stream/) ::: </li>
      </ul>
</div>
  </div>

  <hr>
</div>


<h2 id="description">
 <a href="#/api/device/createStream?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

Creates and returns a new [occa::stream](/api/stream/) to queue operations on.
If the backend supports streams, out-of-order work can be achieved through
the use of streams.

> Note that the stream is created but not set as the active stream.
