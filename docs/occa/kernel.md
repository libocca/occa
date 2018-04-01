Building and Running Kernels
========================================

Now that we have data in the device, we want to start computing our vector addition.
We must first build a kernel that uses device data.

Kernels are built at runtime so we require 2 things

- The file with the kernel source code
- The name of the kernel in the source code we wish to use

.. tabs::

   .. code-tab:: c++

      occa::kernel addVectors = device.buildKernel("addVectors.okl", "addVectors");

   .. code-tab:: c

      occaKernel addVectors = deviceBuildKernel(device, "addVectors.okl", "addVectors");

   .. code-tab:: py

      add_vectors = device.build_kernel("addVectors.okl", "addVectors")

   .. code-tab:: fortran

      type(occaKernel) :: addVectors
      addVectors = occaDeviceBuildKernel(device, "addVectors.okl", "addVectors")

We can now call :code:`addVectors` with our device arrays.

.. tabs::

   .. code-tab:: c++

      addVectors(entries, o_a, o_b, o_ab);

   .. code-tab:: c

      occaKernelRun(addVectors, occaInt32(entries), o_a, o_b, o_ab);

   .. code-tab:: py

      add_vectors(entries, o_a, o_b, o_ab)

   .. code-tab:: fortran

      call occaKernelRun(addVectors, occaInt32(entries), o_a, o_b, o_ab)
