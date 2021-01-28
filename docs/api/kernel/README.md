<h1 id="occa::kernel">
 <a href="#/api/kernel/" class="anchor">
   <span>occa::kernel</span>
  </a>
</h1>

<h2 id="description">
 <a href="#/api/kernel/?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>


A [occa::kernel](/api/kernel/) object is a handle to a device function for the device it was built in.
For example, in `Serial` and `OpenMP` modes it is analogous to a calling a C++ function.
For GPU modes, it means launching work on a more granular and parallized manner.

<h2 id="launch">
 <a href="#/api/kernel/?id=launch" class="anchor">
   <span>Launch</span>
  </a>
</h2>

There are 2 ways to launching kernels:
- [operator ()](/api/kernel/operator_parentheses) which can be used to call a kernel like a regular function.
- [run](/api/kernel/run) which requires the user to push the arguments one-by-one before running it.

<h2 id="garbage collection">
 <a href="#/api/kernel/?id=garbage collection" class="anchor">
   <span>Garbage collection</span>
  </a>
</h2>

The [free](/api/kernel/free) function can be called to free the kernel.
OCCA implemented reference counting by default so calling [free](/api/kernel/free) is not required.