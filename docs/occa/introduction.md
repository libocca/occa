# Introduction

The design of the OCCA (like *oca*-rina) API is based on the functionality of physical devices.
The main two being

- Memory management
- Code execution

We generalize different device architectures by wrapping these two concepts in a single API.
We'll showcase the basics by going through a simple example where we add 2 vectors.

# Terminology

<span style="font-size: 1.2em">_Host_</span>
<template><div style="margin-top: -0.8em; padding-left: 1em;">
The physical device running the application code.
This is usually a CPU processor.
</div></template>

<span style="font-size: 1.2em">_Device_</span>
<template><div style="margin-top: -0.8em; padding-left: 1em;">
A physical device we're communicating with, whether the same physical device as the host or an offload device.
Examples include CPU processors, GPUs, and Xeon Phi.
</div></template>


<span style="font-size: 1.2em">_Kernel_</span>
<template><div style="margin-top: -0.8em; padding-left: 1em;">
A function that runs on a device
</div></template>
