#ifndef OCCA_UTILS_UVA_HEADER
#define OCCA_UTILS_UVA_HEADER

#include <iostream>
#include <vector>

#include <occa/defines.hpp>
#include <occa/types.hpp>

namespace occa {
  occa::modeMemory_t* uvaToMemory(void *ptr);

  bool isManaged(void *ptr);
  void startManaging(void *ptr);
  void stopManaging(void *ptr);

  void syncToDevice(void *ptr, const udim_t bytes = (udim_t) -1);
  void syncToHost(void *ptr, const udim_t bytes = (udim_t) -1);

  void syncMemToDevice(occa::modeMemory_t *mem,
                       const udim_t bytes = (udim_t) -1,
                       const udim_t offset = 0);

  void syncMemToHost(occa::modeMemory_t *mem,
                     const udim_t bytes = (udim_t) -1,
                     const udim_t offset = 0);

  bool needsSync(void *ptr);
  void sync(void *ptr);
  void dontSync(void *ptr);

  void freeUvaPtr(void *ptr);
}

#endif
