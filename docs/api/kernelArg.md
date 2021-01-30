<h1 id="occa::kernel-arg">
 <a href="#/api/kernelArg" class="anchor">
   <span>occa::kernelArg</span>
  </a>
</h1>

<h2 id="description">
 <a href="#/api/kernelArg?id=description" class="anchor">
   <span>Description</span>
  </a>
</h2>

[occa::kernel](/api/kernel/) arguments must be of type [occa::kernelArg](/api/kernelArg).
Custom user types can be passed to a kernel by implementing a cast operator such as:

```cpp
operator occa::kernelArg() const;
```