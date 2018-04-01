Allocating and Initializing Memory
========================================

Now that we have a device, we need to allocate and initialize memory in the device.
We cannot modify device memory directly in the host, hence data is usually initialized either by:

- Copying host data to the device
- Modifying device data inside a kernel running on the device

For our case, we'll initialize data in the host and copy it to the device.

.. tabs::

   .. code-tab:: c++

      const int entries = 5;
      float *a  = new float[entries];
      float *b  = new float[entries];
      float *ab = new float[entries];

      for (int i = 0; i < entries; ++i) {
        a[i]  = i;
        b[i]  = 1 - i;
        ab[i] = 0;
      }

   .. code-tab:: c

      const int entries = 5;
      float *a  = (float*) malloc(entries * sizeof(float));
      float *b  = (float*) malloc(entries * sizeof(float));
      float *ab = (float*) malloc(entries * sizeof(float));

      for (int i = 0; i < entries; ++i) {
        a[i]  = i;
        b[i]  = 1 - i;
        ab[i] = 0;
      }

   .. code-tab:: py

      entries = 5
      a = [i for i in range(entries)]
      b = [1 - i for i in range(entries)]

   .. code-tab:: fortran

      !???

We now allocate and initialize device memory using our host arrays.

We initialize both :code:`o_a` and :code:`o_b` by copying over data from :code:`a` and :code:`b` respectively.
Since we'll be computing :code:`o_ab` by summing :code:`o_a` and :code:`o_b`, there is no need to transfer :code:`ab` to the device.

.. tabs::

   .. code-tab:: c++

      occa::memory o_a  = device.malloc(entries * sizeof(float), a);
      occa::memory o_b  = device.malloc(entries * sizeof(float), b);
      occa::memory o_ab = device.malloc(entries * sizeof(float));

   .. code-tab:: c

      occaMemory o_a  = occaDeviceMalloc(device, entries * sizeof(float),
                                         a, occaDefault);
      occaMemory o_b  = occaDeviceMalloc(device, entries * sizeof(float),
                                         b, occaDefault);
      occaMemory o_ab = occaDeviceMalloc(device, entries * sizeof(float),
                                         NULL, occaDefault);

   .. code-tab:: py

      o_a  = device.malloc(a, dtype=np.float32)
      o_b  = device.malloc(b, dtype=np.float32)
      o_ab = device.malloc(entries, dtype=np.float32)

   .. code-tab:: fortran

      !???
