Unified Memory
========================================

Unified memory, or UM for short, is another tool that can facilitate the addition of OCCA in existing codes.
Rather than working with memory objects, we work with pointers (only available in C++ and C).

.. tabs::

   .. code-tab:: c++

      int *array = (int*) occa::umalloc(10 * sizeof(int));

   .. code-tab:: c

      int *array = (int*) occaUmalloc(10 * sizeof(int));

The resulting pointer indirectly maps to memory in the device.
