Synchronization
========================================

There is no guarantee in the completion of kernel calls.
Hence we have to synchronize the device with the host.

.. tabs::

   .. code-tab:: c++

      device.finish();

   .. code-tab:: c

      occaDeviceFinish(device);

   .. code-tab:: py

      device.finish()

   .. code-tab:: fortran

      call occaDeviceFinish(device)

Alternatively, we can do non-asynchronous (blocking) copies to guarantee all of the device executions have finished.

.. tabs::

   .. code-tab:: c++

      occa::memcpy(b, o_b);

   .. code-tab:: c

      occaCopyMemToPtr(b, o_b, occaAllBytes, 0, occaDefault);

   .. code-tab:: py

      occa.memcpy(b, o_b)

   .. code-tab:: fortran

      call occaCopyMemToPtr(b, o_b, occaAllBytes, 0, occaDefault);

Now we can finally test our results by looking at our host data in :code:`ab`.

.. tabs::

   .. code-tab:: c++

      for (int i = 0; i < entries; ++i) {
        if (b[i] != 1) {
          std::cerr("addVectors failed\n");
        }
      }

   .. code-tab:: c

      for (int i = 0; i < entries; ++i) {
        if (b[i] != 1) {
          fprintf(stderr, "addVectors failed\n");
        }
      }

   .. code-tab:: py

      if not np.array_equal1(b, [1] * entries):
        raise Exception('add_vectors failed')

   .. code-tab:: fortran

      !???
