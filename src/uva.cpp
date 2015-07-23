#include "occa.hpp"

namespace occa {
  ptrRangeMap_t uvaMap;
  memoryVector_t uvaDirtyMemory;

  bool hasUvaEnabledByDefault(){
    return uvaEnabledByDefault_f;
  }

  void enableUvaByDefault(){
    uvaEnabledByDefault_f = true;
  }

  void disableUvaByDefault(){
    uvaEnabledByDefault_f = false;
  }

  ptrRange_t::ptrRange_t() :
    start(NULL),
    end(NULL) {}

  ptrRange_t::ptrRange_t(void *ptr, const uintptr_t bytes) :
    start((char*) ptr),
    end(((char*) ptr) + bytes) {}

  ptrRange_t::ptrRange_t(const ptrRange_t &r) :
    start(r.start),
    end(r.end) {}

  ptrRange_t& ptrRange_t::operator = (const ptrRange_t &r){
    start = r.start;
    end   = r.end;

    return *this;
  }

  bool ptrRange_t::operator == (const ptrRange_t &r) const {
    return ((start <= r.start) && (r.start < end));
  }

  bool ptrRange_t::operator != (const ptrRange_t &r) const {
    return ((r.start < start) || (end <= r.start));
  }

  int operator < (const ptrRange_t &a, const ptrRange_t &b){
    return ((a != b) && (a.start < b.start));
  }

  uvaPtrInfo_t::uvaPtrInfo_t() :
    mem(NULL) {}

  uvaPtrInfo_t::uvaPtrInfo_t(void *ptr){
    ptrRangeMap_t::iterator it = uvaMap.find(ptr);

    if(it != uvaMap.end())
      mem = (it->second);
    else
      mem = (occa::memory_v*) ptr; // Defaults to ptr being a memory_v
  }

  uvaPtrInfo_t::uvaPtrInfo_t(occa::memory_v *mem_) :
    mem(mem_) {}

  uvaPtrInfo_t::uvaPtrInfo_t(const uvaPtrInfo_t &upi) :
    mem(upi.mem) {}

  uvaPtrInfo_t& uvaPtrInfo_t::operator = (const uvaPtrInfo_t &upi){
    mem = upi.mem;

    return *this;
  }

  occa::device uvaPtrInfo_t::getDevice(){
    occa::memory m(mem);

    return occa::device(m.getDHandle());
  }

  occa::memory uvaPtrInfo_t::getMemory(){
    return occa::memory(mem);
  }

  occa::memory_v* uvaToMemory(void *ptr){
    ptrRangeMap_t::iterator it = uvaMap.find(ptr);

    if(it == uvaMap.end())
      return NULL;

    return it->second;
  }

  void startManaging(void *ptr){
    occa::memory_v *mem = uvaToMemory(ptr);

    if(mem == NULL)
      return;

    mem->memInfo &= ~uvaFlag::leftInDevice;
  }

  void stopManaging(void *ptr){
    occa::memory_v *mem = uvaToMemory(ptr);

    if(mem == NULL)
      return;

    mem->memInfo |= uvaFlag::leftInDevice;
  }

  void syncToDevice(void *ptr, const uintptr_t bytes){
    occa::memory_v *mem = uvaToMemory(ptr);

    if(mem == NULL)
      return;

    syncMemToDevice(mem, bytes, ptrDiff(mem->uvaPtr, ptr));
  }

  void syncFromDevice(void *ptr, const uintptr_t bytes){
    occa::memory_v *mem = uvaToMemory(ptr);

    if(mem == NULL)
      return;

    syncMemFromDevice(mem, bytes, ptrDiff(mem->uvaPtr, ptr));
  }

  void syncMemToDevice(occa::memory_v *mem,
                       const uintptr_t bytes,
                       const uintptr_t offset){

    if(!mem->dHandle->fakesUva()){
      memcpy(occa::memory(mem->handle),
             ptrOff(mem->uvaPtr, offset),
             bytes, offset);
    }
    else {
      occa::memory(mem).syncToDevice(bytes, offset);
    }
  }

  void syncMemFromDevice(occa::memory_v *mem,
                         const uintptr_t bytes,
                         const uintptr_t offset){

    if(!mem->dHandle->fakesUva()){
      memcpy(ptrOff(mem->uvaPtr, offset),
             occa::memory(mem->handle),
             bytes, offset);
    }
    else {
      occa::memory(mem).syncFromDevice(bytes, offset);
    }
  }

  bool needsSync(void *ptr){
    occa::memory_v *mem = uvaToMemory(ptr);

    if(mem == NULL)
      return false;

    return mem->isDirty();
  }

  void sync(void *ptr){
    occa::memory_v *mem = uvaToMemory(ptr);

    if(mem == NULL)
      return;

    if(mem->inDevice())
      syncMemFromDevice(mem);
    else
      syncMemToDevice(mem);
  }

  void dontSync(void *ptr){
    removeFromDirtyMap(ptr);
  }

  void removeFromDirtyMap(void *ptr){
    ptrRangeMap_t::iterator it = uvaMap.find(ptr);

    if(it == uvaMap.end())
      return;

    memory m(it->second);

    if(!m.uvaIsDirty())
      return;

    removeFromDirtyMap(m.getMHandle());
  }

  void removeFromDirtyMap(memory_v *mem){
    occa::memory m(mem);

    const size_t dirtyEntries = uvaDirtyMemory.size();

    for(size_t i = 0; i < dirtyEntries; ++i){
      if(uvaDirtyMemory[i] == mem){
        m.uvaMarkClean();
        uvaDirtyMemory.erase(uvaDirtyMemory.begin() + i);

        break;
      }
    }
  }

  void setupMagicFor(void *ptr){
    ptrRangeMap_t::iterator it = uvaMap.find(ptr);

    if(it == uvaMap.end())
      return;

    memory_v &mem = *(it->second);

    if(mem.dHandle->fakesUva())
      return;

    if(mem.uvaPtr == NULL)
      mem.uvaPtr = cpu::malloc(mem.size);

    memcpy(mem.uvaPtr, mem.handle, mem.size);
  }

  void free(void *ptr){
    ptrRangeMap_t::iterator it = uvaMap.find(ptr);

    if((it != uvaMap.end()) &&
       (((void*) it->first.start) != ((void*) it->second))){

      occa::memory(it->second).free();
    }
    else
      ::free(ptr);
  }
}
