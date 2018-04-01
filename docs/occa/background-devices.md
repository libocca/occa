Background Devices
========================================

Most applications might only use one device, thus we provide a simpler API for such cases.
Similar to CUDA, there is a *background device* that can be used to call methods that would normally require an actual device object.

For example, when allocating memory

.. tabs::

   .. group-tab:: C++

      .. code-block:: c++

         device.malloc(10 * sizeof(int));

      would turn into

      .. code-block:: c++

         occa::malloc(10 * sizeof(int));

   .. group-tab:: C

      .. code-block:: c

         occaDeviceMalloc(device, 10 * sizeof(int));

      would turn into

      .. code-block:: c

         occaMalloc(10 * sizeof(int));

   .. group-tab:: Python

      .. code-block:: py

         device.malloc(10 * sizeof(int))

      would turn into

      .. code-block:: py

         occa.malloc(10 * sizeof(int))

   .. group-tab:: Fortran

      .. code-block:: fortran

         !???

      would turn into

      .. code-block:: fortran

         !???

The default background device is set to

.. tabs::

   .. code-tab:: c++

      occa::host()

   .. code-tab:: c

      occaHost()

   .. code-tab:: py

      occa.host

   .. code-tab:: fortran

      !???

Methods to fetch and set the background device are

.. tabs::

   .. code-tab:: c++

      occa::setDevice(device);
      occa::setDevice("mode: 'Serial'");

   .. code-tab:: c

      occaSetDevice(device);
      occaSetDevice(occaString("mode: 'Serial'"));

   .. code-tab:: py

      occa.set_device(device)
      occa.set_device("mode: 'Serial'")

   .. code-tab:: fortran

      !???

A powerful use of the background device is the ability to easy the inclusion of OCCA in existing libraries.
Changing API is not always an easy process, specially when adding a library such as OCCA that targets the most computational intensive parts of the code.
Having the background device implicitly allows a device to be used inside methods without passing it as a function argument.
