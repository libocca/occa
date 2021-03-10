<h1 id="occa::memory">
 <a href="#/api/memory/" class="anchor">
   <span>occa::memory</span>
  </a>
</h1>

<h2 id="description">
 <a href="#/api/memory/?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>


A [occa::memory](/api/memory/) object is a handle to memory allocated by a device.
For example, in `Serial` and `OpenMP` modes it is analogous to a `void*` pointer that comes out of `malloc` or `new`.
Check [malloc](/api/device/malloc) for more information about how to allocate memory and build a memory object.

<h2 id="data transfer">
 <a href="#/api/memory/?id=data transfer" class="anchor">
   <span>Data transfer</span>
  </a>
</h2>

There are 2 helper methods to help with data transfer:
- [copyTo](/api/memory/copyTo) which helpes copy data from the memory object to the input.
- [copyFrom](/api/memory/copyFrom) which helpes copy data from the input to the memory object.

> Note that because we know the type and size of the underlying data allocated, passing the bytes to copy defaults to the full array.

<h2 id="transformations">
 <a href="#/api/memory/?id=transformations" class="anchor">
   <span>Transformations</span>
  </a>
</h2>

<h3 id="slices">
 <a href="#/api/memory/?id=slices" class="anchor">
   <span>Slices</span>
  </a>
</h3>

Sometimes we want to pass a subsection of the memory to a kernel.
Rather than passing the memory and the offset to the kernel, we support slicing the memory object through [slice](/api/memory/slice).
The returned memory object will be a reference to the original but will keep track of the offset and size change.

<h3 id="cloning">
 <a href="#/api/memory/?id=cloning" class="anchor">
   <span>Cloning</span>
  </a>
</h3>

The [clone](/api/memory/clone) method is a quick way to create a copy of a memory object.

<h3 id="casting">
 <a href="#/api/memory/?id=casting" class="anchor">
   <span>Casting</span>
  </a>
</h3>

Calling [cast](/api/memory/cast) will return a reference to the original memory object but with a different type.
This can be used to assert type at runtime when passed to kernel as arguments.

<h2 id="garbage collection">
 <a href="#/api/memory/?id=garbage collection" class="anchor">
   <span>Garbage collection</span>
  </a>
</h2>

The [free](/api/memory/free) function can be called to free the memory.
OCCA implemented reference counting by default so calling [free](/api/memory/free) is not required.