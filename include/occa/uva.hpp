#include <iostream>
#include <vector>

#include "occa/defines.hpp"

namespace occa {
  typedef int mode;

  class kernel_v;
  template <occa::mode> class kernel_t;
  class kernel;

  class memory_v;
  template <occa::mode> class memory_t;
  class memory;

  class device_v;
  template <occa::mode> class device_t;
  class device;

  bool hasUvaEnabledByDefault();
  void enableUvaByDefault();
  void disableUvaByDefault();

  class ptrRange_t {
  public:
    char *start, *end;

    ptrRange_t();
    ptrRange_t(void *ptr, const uintptr_t bytes = 0);
    ptrRange_t(const ptrRange_t &r);

    ptrRange_t& operator =  (const ptrRange_t &r);
    bool        operator == (const ptrRange_t &r) const;
    bool        operator != (const ptrRange_t &r) const;

    friend int operator < (const ptrRange_t &a, const ptrRange_t &b);
  };

  typedef std::map<ptrRange_t, occa::memory_v*> ptrRangeMap_t;
  typedef std::vector<occa::memory_v*>          memoryVector_t;

  extern ptrRangeMap_t uvaMap;
  extern memoryVector_t uvaDirtyMemory;

  class uvaPtrInfo_t {
  private:
    occa::memory_v *mem;

  public:
    uvaPtrInfo_t();
    uvaPtrInfo_t(void *ptr);
    uvaPtrInfo_t(occa::memory_v *mem_);

    uvaPtrInfo_t(const uvaPtrInfo_t &upi);
    uvaPtrInfo_t& operator = (const uvaPtrInfo_t &upi);

    occa::device getDevice();
    occa::memory getMemory();
  };

  occa::memory_v* uvaToMemory(void *ptr);

  void startManaging(void *ptr);
  void stopManaging(void *ptr);

  void syncToDevice(void *ptr, const uintptr_t bytes = 0);
  void syncFromDevice(void *ptr, const uintptr_t bytes = 0);

  void syncMemToDevice(occa::memory_v *mem,
                       const uintptr_t bytes = 0,
                       const uintptr_t offset = 0);

  void syncMemFromDevice(occa::memory_v *mem,
                         const uintptr_t bytes = 0,
                         const uintptr_t offset = 0);

  bool needsSync(void *ptr);
  void sync(void *ptr);
  void dontSync(void *ptr);

  void removeFromDirtyMap(void *ptr);
  void removeFromDirtyMap(memory_v *mem);

  void setupMagicFor(void *ptr);

  void free(void *ptr);
}
