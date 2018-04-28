<img
    src="./assets/images/logo/blue.svg"
    width="250"
    style="display: block; width: 250px; margin: auto; margin-bottom: 3em"
/>

OCCA (like *oca*-rina) is an open-source library that facilitates programming in an environment containing different types of devices.
We abstract devices and let the user pick at run-time, for example: CPUs, GPUs, Intel's Xeon Phi, FPGAs.

We also provide OKL, a kernel programming language that minimally extends C and restricts the user to write parallel code. The code is transformed into the corresponding backend, such as OpenMP, OpenCL, or CUDA, and JIT compiled.
