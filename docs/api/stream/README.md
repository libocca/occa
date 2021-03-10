<h1 id="occa::stream">
 <a href="#/api/stream/" class="anchor">
   <span>occa::stream</span>
  </a>
</h1>

<h2 id="description">
 <a href="#/api/stream/?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

A [occa::device](/api/device/) has one active [occa::stream](/api/stream/) at a time.
If the backend supports it, using multiple streams will achieve better parallelism by having more work queued up.
Work on a stream is considered to be done in order, but can be out of order if work is queued using multiple streams.