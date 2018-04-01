<div style="width: 200px; margin: auto; margin-bottom: 3em">
![OCCA](_images/blue-logo.png)
</div>

OCCA is an open-source library that facilitates programming in an environment containing different types of devices.
We abstract devices and let the user pick at run-time, for example: CPUs, GPUs, Intel's Xeon Phi, FPGAs.

We abstract the device programming langauges into one kernel language, the OCCA kernel language (OKL).
OKL minimally extends C and restricts the user to write parallel code that JIT compiled.
