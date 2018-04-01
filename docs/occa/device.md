Connecting to a Device
========================================

We start off by connecting to a physical device through an OCCA device object.

.. tabs::

   .. code-tab:: c++

      #include "occa.hpp"

      occa::device device("mode: 'Serial'");

   .. code-tab:: c

      #include "occa_c.h"

      occaDevice device = occaCreateDevice("mode: 'Serial'");

   .. code-tab:: py

      import occa
      import np

      device = occa.Device("mode: 'Serial'")

   .. code-tab:: fortran

      use occa

      type(occaDevice) :: device
      device = occaCreateDevice("mode: 'Serial'")

The initialization string passed to the device constructor

.. code-block:: c++

   "mode: 'Serial'"

creates a `properties <../api/properties.html>`_ object.
Properties are handled as a JSON object, using shorthand notations found in JavaScript.

.. code-block:: js

   {
     mode: 'Serial'
   }

The only property field required by OCCA when creating a device is :code:`mode`.
However, each :code:`mode` has its own requirements such as :code:`deviceID` for CUDA.
In this case, we're initializing a device that runs code serially (usually useful for debugging).

Here are examples for the all core modes supported in OCCA.

.. tabs::

   .. tab:: Serial

      .. code-block:: c++

         "mode: 'Serial'"

      .. code-block:: js

         {
           mode: 'Serial'
         }

   .. tab:: OpenMP

      .. code-block:: c++

         "mode: 'OpenMP', threads: 4"

      .. code-block:: js

         {
           mode: 'OpenMP',
           threads: 4
         }

   .. tab:: Threads

      .. code-block:: c++

         "mode: 'Serial', threads: 4, pinnedCores: [0, 1, 2, 3]"

      .. code-block:: js

         {
           mode: 'Serial',
           threads: 4,
           pinnedCores: [0, 1, 2, 3]
         }

   .. tab:: OpenCL

      .. code-block:: c++

         "mode: 'OpenCL', deviceID: 0, platformID: 0"

      .. code-block:: js

         {
           mode: 'OpenCL',
           deviceID: 0,
           platformID: 0
         }

   .. tab:: CUDA

      .. code-block:: c++

         "mode: 'CUDA', deviceID: 0"

      .. code-block:: js

         {
           mode: 'CUDA',
           deviceID: 0
         }
