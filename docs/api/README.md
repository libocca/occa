# API

Welcome to the OCCA library API documentation!


Below are the more common class types but all of the documentation can be found on the left sidebar.
On mobile, expand the sidebar through the bottom-right button.

[occa::device](/api/device/)

::: indent
A [occa::device](/api/device/) object maps to a physical device we want to program on.
Examples include a CPU, GPU, or other physical accelerator like an FPGA.
:::

[occa::kernel](/api/kernel/)

::: indent
A [occa::kernel](/api/kernel/) object is a handle to a device function for the device it was built in.
For example, in `Serial` and `OpenMP` modes it is analogous to a calling a C++ function.
For GPU modes, it means launching work on a more granular and parallized manner.
:::

[occa::memory](/api/memory/)

::: indent
A [occa::memory](/api/memory/) object is a handle to memory allocated by a device.
:::

[occa::json](/api/json/)

::: indent
A [occa::json](/api/json/) object stores data in the same way specified by the JSON standard.
It's used across the OCCA library as a way to flexibly pass user configurations.
:::
