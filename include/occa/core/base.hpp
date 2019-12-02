#ifndef OCCA_CORE_BASE_HEADER
#define OCCA_CORE_BASE_HEADER

#include <iostream>
#include <vector>

#include <stdint.h>

#include <occa/defines.hpp>
#include <occa/types.hpp>

#if OCCA_SSE
#  include <xmmintrin.h>
#endif

#include <occa/core/device.hpp>
#include <occa/core/kernel.hpp>
#include <occa/core/memory.hpp>
#include <occa/core/stream.hpp>
#include <occa/core/streamTag.hpp>

namespace occa {
  //---[ Device Functions ]-------------
  device host();
  device& getDevice();

  void setDevice(device d);
  void setDevice(const occa::properties &props);

  const occa::properties& deviceProperties();

  void loadKernels(const std::string &library = "");

  void finish();

  void waitFor(streamTag tag);

  double timeBetween(const streamTag &startTag,
                     const streamTag &endTag);

  stream createStream(const occa::properties &props = occa::properties());
  stream getStream();
  void setStream(stream s);

  streamTag tagStream();
  //====================================

  //---[ Kernel Functions ]-------------
  kernel buildKernel(const std::string &filename,
                     const std::string &kernelName,
                     const occa::properties &props = occa::properties());

  kernel buildKernelFromString(const std::string &content,
                               const std::string &kernelName,
                               const occa::properties &props = occa::properties());

  kernel buildKernelFromBinary(const std::string &filename,
                               const std::string &kernelName,
                               const occa::properties &props = occa::properties());
  //====================================

  //---[ Memory Functions ]-------------
  occa::memory malloc(const dim_t entries,
                      const dtype_t &dtype,
                      const void *src = NULL,
                      const occa::properties &props = occa::properties());

  template <class TM = void>
  occa::memory malloc(const dim_t entries,
                      const void *src = NULL,
                      const occa::properties &props = occa::properties());

  template <>
  occa::memory malloc<void>(const dim_t entries,
                            const void *src,
                            const occa::properties &props);

  void* umalloc(const dim_t entries,
                const dtype_t &dtype,
                const void *src = NULL,
                const occa::properties &props = occa::properties());

  template <class TM = void>
  TM* umalloc(const dim_t entries,
              const void *src = NULL,
              const occa::properties &props = occa::properties());

  template <>
  void* umalloc<void>(const dim_t entries,
                      const void *src,
                      const occa::properties &props);

  void memcpy(void *dest, const void *src,
              const dim_t bytes,
              const occa::properties &props = properties());

  void memcpy(memory dest, const void *src,
              const dim_t bytes = -1,
              const dim_t offset = 0,
              const occa::properties &props = properties());

  void memcpy(void *dest, memory src,
              const dim_t bytes = -1,
              const dim_t offset = 0,
              const occa::properties &props = properties());

  void memcpy(memory dest, memory src,
              const dim_t bytes = -1,
              const dim_t destOffset = 0,
              const dim_t srcOffset = 0,
              const occa::properties &props = properties());

  void memcpy(void *dest, const void *src,
              const occa::properties &props);

  void memcpy(memory dest, const void *src,
              const occa::properties &props);

  void memcpy(void *dest, memory src,
              const occa::properties &props);

  void memcpy(memory dest, memory src,
              const occa::properties &props);

  namespace cpu {
    occa::memory wrapMemory(void *ptr,
                            const udim_t bytes,
                            const occa::properties &props = occa::properties());
  }
  //====================================

  //---[ Free Functions ]---------------
  void free(device d);
  void free(stream s);
  void free(kernel k);
  void free(memory m);
  //====================================

  void printModeInfo();
}

#include "base.tpp"

#endif
