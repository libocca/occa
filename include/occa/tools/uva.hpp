#ifndef OCCA_UVA_HEADER
#define OCCA_UVA_HEADER

#include <iostream>
#include <vector>

#include <occa/defines.hpp>
#include <occa/io/output.hpp>
#include <occa/types.hpp>

namespace occa {
  class device;
  class memory;
  class modeMemory_t;
  class ptrRange;

  typedef std::map<ptrRange, occa::modeMemory_t*> ptrRangeMap;
  typedef std::vector<occa::modeMemory_t*>        memoryVector;

  extern ptrRangeMap uvaMap;
  extern memoryVector uvaStaleMemory;

  //---[ ptrRange ]---------------------
  class ptrRange {
  public:
    char *start, *end;

    ptrRange();
    ptrRange(void *ptr, const udim_t bytes = 0);
    ptrRange(const ptrRange &other);

    ptrRange& operator =  (const ptrRange &other);
    bool operator == (const ptrRange &other) const;
    bool operator != (const ptrRange &other) const;
  };

  int operator < (const ptrRange &a,
                  const ptrRange &b);

  std::ostream& operator << (std::ostream& out,
                           const ptrRange &range);
  //====================================


  //---[ UVA ]--------------------------
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

  void removeFromStaleMap(void *ptr);
  void removeFromStaleMap(modeMemory_t *mem);

  void freeUvaPtr(void *ptr);
  //====================================
}

#endif
