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
  void setDevice(const std::string &props);
  void setDevice(const occa::json &props);
  void setDevice(jsonInitializerList initializer);

  const occa::json& deviceProperties();

  void finish();

  void waitFor(streamTag tag);

  double timeBetween(const streamTag &startTag,
                     const streamTag &endTag);

  stream createStream(const occa::json &props = occa::json());
  stream getStream();
  void setStream(stream s);

  streamTag tagStream();

  experimental::memoryPool createMemoryPool(const occa::json &props = occa::json());
  //====================================

  //---[ Kernel Functions ]-------------
  kernel buildKernel(const std::string &filename,
                     const std::string &kernelName,
                     const occa::json &props = occa::json());

  kernel buildKernelFromString(const std::string &content,
                               const std::string &kernelName,
                               const occa::json &props = occa::json());

  kernel buildKernelFromBinary(const std::string &filename,
                               const std::string &kernelName,
                               const occa::json &props = occa::json());
  //====================================

  //---[ Memory Functions ]-------------
  occa::memory malloc(const dim_t entries,
                      const dtype_t &dtype,
                      const void *src = NULL,
                      const occa::json &props = occa::json());

  template <class T = void>
  occa::memory malloc(const dim_t entries,
                      const void *src = NULL,
                      const occa::json &props = occa::json());

  template <>
  occa::memory malloc<void>(const dim_t entries,
                            const void *src,
                            const occa::json &props);

  void memcpy(memory dest, const void *src,
              const dim_t bytes = -1,
              const dim_t offset = 0,
              const occa::json &props = json());

  void memcpy(void *dest, memory src,
              const dim_t bytes = -1,
              const dim_t offset = 0,
              const occa::json &props = json());

  void memcpy(memory dest, memory src,
              const dim_t bytes = -1,
              const dim_t destOffset = 0,
              const dim_t srcOffset = 0,
              const occa::json &props = json());

  void memcpy(memory dest, const void *src,
              const occa::json &props);

  void memcpy(void *dest, memory src,
              const occa::json &props);

  void memcpy(memory dest, memory src,
              const occa::json &props);

  occa::memory wrapMemory(const void *ptr,
                          const dim_t entries,
                          const dtype_t &dtype,
                          const occa::json &props = occa::json());

  template <class T = void>
  occa::memory wrapMemory(const T *ptr,
                          const dim_t entries,
                          const occa::json &props = occa::json());

  template <>
  occa::memory wrapMemory<void>(const void *ptr,
                                const dim_t entries,
                                const occa::json &props);
  //====================================

  //---[ Free Functions ]---------------
  void free(device d);
  void free(stream s);
  void free(kernel k);
  void free(memory m);
  //====================================

  //---[ Helper Methods ]---------------
  bool modeIsEnabled(const std::string &mode);

  int getDeviceCount(const std::string &props);
  int getDeviceCount(const occa::json &props);

  void printModeInfo();
  //====================================

  template<typename T>
  void* unwrap(T& occaType);
}

#include "base.tpp"

#endif
