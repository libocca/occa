<h1 id="occa::stream-tag">
 <a href="#/api/streamTag/" class="anchor">
   <span>occa::streamTag</span>
  </a>
</h1>

<h2 id="description">
 <a href="#/api/streamTag/?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

The result from calling [tagStream](/api/device/tagStream).

A stream tag can be used to check how much time elapsed between two tags ([timeBetween](/api/device/timeBetween)).

A stream tag can also be used to wait all work queued up before the tag ([streamTag.wait()](/api/streamTag/wait) or [device.waitFor()](/api/device/waitFor)).