
<h1 id="time-between">
 <a href="#/api/device/timeBetween" class="anchor">
   <span>timeBetween</span>
  </a>
</h1>

<div class="signature">

<hr>

  <div class="definition-container">
    <div class="definition">
      <code class="desktop-only"><span class="token keyword">double</span> timeBetween(<span class="token keyword">const</span> <a href="#/api/streamTag/">occa::streamTag</a> &amp;startTag,
                   <span class="token keyword">const</span> <a href="#/api/streamTag/">occa::streamTag</a> &amp;endTag)</code>
      <code class="mobile-only"><span class="token keyword">double</span> timeBetween(
    <span class="token keyword">const</span> <a href="#/api/streamTag/">occa::streamTag</a> &amp;startTag,
    <span class="token keyword">const</span> <a href="#/api/streamTag/">occa::streamTag</a> &amp;endTag
)</code>
      <div class="flex-spacing"></div>
      <a href="https://github.com/libocca/occa/blob/6d155d0c/include/occa/core/device.hpp#L436" target="_blank">Source</a>
    </div>
    <div class="description">

      <div class="section-header">Returns</div>
      <ul class="section-list">
        <li> ::: markdown Returns the time in seconds. ::: </li>
      </ul>
</div>
  </div>

  <hr>
</div>


<h2 id="description">
 <a href="#/api/device/timeBetween?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

Returns the time taken between two [tags](/api/streamTag/).
