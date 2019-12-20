#include <map>

#include <occa/core/base.hpp>
#include <occa/core/memory.hpp>
#include <occa/core/device.hpp>
#include <occa/modes/serial/memory.hpp>
#include <occa/tools/uva.hpp>
#include <occa/tools/sys.hpp>

namespace occa {
  //---[ modeMemory_t ]-----------------
  modeMemory_t::modeMemory_t(modeDevice_t *modeDevice_,
                             udim_t size_,
                             const occa::properties &properties_) :
    memInfo(uvaFlag::none),
    properties(properties_),
    ptr(NULL),
    uvaPtr(NULL),
    modeDevice(modeDevice_),
    dtype_(&dtype::byte),
    size(size_),
    isOrigin(true) {
    modeDevice->addMemoryRef(this);
  }

  modeMemory_t::~modeMemory_t() {
    // NULL all wrappers
    while (memoryRing.head) {
      memory *mem = (memory*) memoryRing.head;
      memoryRing.removeRef(mem);
      mem->modeMemory = NULL;
    }
    // Remove ref from device
    if (modeDevice) {
      modeDevice->removeMemoryRef(this);
    }
  }

  void* modeMemory_t::getPtr(const occa::properties &props) {
    return ptr;
  }

  void modeMemory_t::dontUseRefs() {
    memoryRing.dontUseRefs();
  }

  void modeMemory_t::addMemoryRef(memory *mem) {
    memoryRing.addRef(mem);
  }

  void modeMemory_t::removeMemoryRef(memory *mem) {
    memoryRing.removeRef(mem);
  }

  bool modeMemory_t::needsFree() const {
    return memoryRing.needsFree();
  }

  bool modeMemory_t::isManaged() const {
    return (memInfo & uvaFlag::isManaged);
  }

  bool modeMemory_t::inDevice() const {
    return (memInfo & uvaFlag::inDevice);
  }

  bool modeMemory_t::isStale() const {
    return (memInfo & uvaFlag::isStale);
  }
  //====================================


  //---[ memory ]-----------------------
  memory::memory() :
      modeMemory(NULL) {}

  memory::memory(void *uvaPtr) :
      modeMemory(NULL) {
    ptrRangeMap::iterator it = uvaMap.find(uvaPtr);
    if (it != uvaMap.end()) {
      setModeMemory(it->second);
    } else {
      setModeMemory((modeMemory_t*) uvaPtr);
    }
  }

  memory::memory(modeMemory_t *modeMemory_) :
      modeMemory(NULL) {
    setModeMemory(modeMemory_);
  }

  memory::memory(const memory &m) :
    modeMemory(NULL) {
    setModeMemory(m.modeMemory);
  }

  memory& memory::operator = (const memory &m) {
    setModeMemory(m.modeMemory);
    return *this;
  }

  memory::~memory() {
    removeMemoryRef();
  }

  void memory::assertInitialized() const {
    OCCA_ERROR("Memory not initialized or has been freed",
               modeMemory != NULL);
  }

  void memory::setModeMemory(modeMemory_t *modeMemory_) {
    if (modeMemory != modeMemory_) {
      removeMemoryRef();
      modeMemory = modeMemory_;
      if (modeMemory) {
        modeMemory->addMemoryRef(this);
      }
    }
  }

  void memory::removeMemoryRef() {
    if (!modeMemory) {
      return;
    }
    modeMemory->removeMemoryRef(this);
    if (modeMemory->modeMemory_t::needsFree()) {
      free();
    }
  }

  void memory::dontUseRefs() {
    if (modeMemory) {
      modeMemory->modeMemory_t::dontUseRefs();
    }
  }

  bool memory::isInitialized() const {
    return (modeMemory != NULL);
  }

  memory& memory::swap(memory &m) {
    modeMemory_t *modeMemory_ = modeMemory;
    modeMemory   = m.modeMemory;
    m.modeMemory = modeMemory_;
    return *this;
  }

  template <>
  void* memory::ptr<void>() {
    return (modeMemory
            ? modeMemory->ptr
            : NULL);
  }

  template <>
  const void* memory::ptr<void>() const {
    return (modeMemory
            ? modeMemory->ptr
            : NULL);
  }

  template <>
  void* memory::ptr<void>(const occa::properties &props) {
    return (modeMemory
            ? modeMemory->getPtr(props)
            : NULL);
  }

  template <>
  const void* memory::ptr<void>(const occa::properties &props) const {
    return (modeMemory
            ? modeMemory->getPtr(props)
            : NULL);
  }

  modeMemory_t* memory::getModeMemory() const {
    return modeMemory;
  }

  modeDevice_t* memory::getModeDevice() const {
    return modeMemory->modeDevice;
  }

  occa::device memory::getDevice() const {
    return occa::device(modeMemory
                        ? modeMemory->modeDevice
                        : NULL);
  }

  memory::operator kernelArg() const {
    if (modeMemory) {
        return modeMemory->makeKernelArg();
    }
    return nullKernelArg;
  }

  const std::string& memory::mode() const {
    static const std::string noMode = "No Mode";
    return (modeMemory
            ? modeMemory->modeDevice->mode
            : noMode);
  }

  const occa::properties& memory::properties() const {
    static const occa::properties noProperties;
    return (modeMemory
            ? modeMemory->properties
            : noProperties);
  }

  void memory::setDtype(const dtype_t &dtype__) {
    assertInitialized();
    OCCA_ERROR("Memory dtype [" << dtype__.name() << "] must be registered",
               dtype__.isRegistered());
    modeMemory->dtype_ = &(dtype__.self());
  }

  const dtype_t& memory::dtype() const {
    if (modeMemory) {
      return *(modeMemory->dtype_);
    }
    return dtype::none;
  }

  udim_t memory::size() const {
    if (modeMemory == NULL) {
      return 0;
    }
    return modeMemory->size;
  }

  udim_t memory::length() const {
    if (modeMemory == NULL) {
      return 0;
    }
    return modeMemory->size / modeMemory->dtype_->bytes();
  }

  bool memory::isManaged() const {
    return (modeMemory && modeMemory->isManaged());
  }

  bool memory::inDevice() const {
    return (modeMemory && modeMemory->inDevice());
  }

  bool memory::isStale() const {
    return (modeMemory && modeMemory->isStale());
  }

  void memory::setupUva() {
    if (!modeMemory) {
      return;
    }
    if ( !(modeMemory->modeDevice->hasSeparateMemorySpace()) ) {
      modeMemory->uvaPtr = modeMemory->ptr;
    } else {
      modeMemory->uvaPtr = (char*) sys::malloc(modeMemory->size);
    }

    ptrRange range;
    range.start = modeMemory->uvaPtr;
    range.end   = (range.start + modeMemory->size);

    uvaMap[range] = modeMemory;
    modeMemory->modeDevice->uvaMap[range] = modeMemory;

    // Needed for kernelArg.void_ -> modeMemory checks
    if (modeMemory->uvaPtr != modeMemory->ptr) {
      uvaMap[modeMemory->ptr] = modeMemory;
    }
  }

  void memory::startManaging() {
    if (modeMemory) {
      modeMemory->memInfo |= uvaFlag::isManaged;
    }
  }

  void memory::stopManaging() {
    if (modeMemory) {
      modeMemory->memInfo &= ~uvaFlag::isManaged;
    }
  }

  void memory::syncToDevice(const dim_t bytes,
                            const dim_t offset) {
    assertInitialized();

    udim_t bytes_ = ((bytes == -1) ? modeMemory->size : bytes);

    OCCA_ERROR("Trying to copy negative bytes (" << bytes << ")",
               bytes >= -1);
    OCCA_ERROR("Cannot have a negative offset (" << offset << ")",
               offset >= 0);

    if (bytes_ == 0) {
      return;
    }

    OCCA_ERROR("Memory has size [" << modeMemory->size << "],"
               << " trying to access [" << offset << ", " << (offset + bytes_) << "]",
               (bytes_ + offset) <= modeMemory->size);

    if (!modeMemory->modeDevice->hasSeparateMemorySpace()) {
      return;
    }

    copyFrom(modeMemory->uvaPtr, bytes_, offset);

    modeMemory->memInfo |=  uvaFlag::inDevice;
    modeMemory->memInfo &= ~uvaFlag::isStale;

    removeFromStaleMap(modeMemory);
  }

  void memory::syncToHost(const dim_t bytes,
                          const dim_t offset) {
    assertInitialized();

    udim_t bytes_ = ((bytes == -1) ? modeMemory->size : bytes);

    OCCA_ERROR("Trying to copy negative bytes (" << bytes << ")",
               bytes >= -1);
    OCCA_ERROR("Cannot have a negative offset (" << offset << ")",
               offset >= 0);

    if (bytes_ == 0) {
      return;
    }

    OCCA_ERROR("Memory has size [" << modeMemory->size << "],"
               << " trying to access [" << offset << ", " << (offset + bytes_) << "]",
               (bytes_ + offset) <= modeMemory->size);

    if (!modeMemory->modeDevice->hasSeparateMemorySpace()) {
      return;
    }

    copyTo(modeMemory->uvaPtr, bytes_, offset);

    modeMemory->memInfo &= ~uvaFlag::inDevice;
    modeMemory->memInfo &= ~uvaFlag::isStale;

    removeFromStaleMap(modeMemory);
  }

  bool memory::uvaIsStale() const {
    return (modeMemory && modeMemory->isStale());
  }

  void memory::uvaMarkStale() {
    if (modeMemory != NULL) {
      modeMemory->memInfo |= uvaFlag::isStale;
    }
  }

  void memory::uvaMarkFresh() {
    if (modeMemory != NULL) {
      modeMemory->memInfo &= ~uvaFlag::isStale;
    }
  }

  bool memory::operator == (const occa::memory &other) const {
    return (modeMemory == other.modeMemory);
  }

  bool memory::operator != (const occa::memory &other) const {
    return (modeMemory != other.modeMemory);
  }

  occa::memory memory::operator + (const dim_t offset) const {
    return slice(offset);
  }

  occa::memory& memory::operator += (const dim_t offset) {
    *this = slice(offset);
    return *this;
  }

  occa::memory memory::slice(const dim_t offset,
                             const dim_t count) const {
    assertInitialized();

    const int dtypeSize = modeMemory->dtype_->bytes();
    const dim_t offset_ = dtypeSize * offset;
    const udim_t bytes  = dtypeSize * ((count == -1)
                                       ? (length() - offset)
                                       : count);

    OCCA_ERROR("Trying to allocate negative bytes (" << bytes << ")",
               bytes >= 0);

    OCCA_ERROR("Cannot have a negative offset (" << offset_ << ")",
               offset_ >= 0);

    OCCA_ERROR("Cannot have offset and bytes greater than the memory size ("
               << offset_ << " + " << bytes << " > " << size() << ")",
               (offset_ + (dim_t) bytes) <= (dim_t) size());

    occa::memory m(modeMemory->addOffset(offset_));
    m.setDtype(dtype());

    modeMemory_t &mm = *(m.modeMemory);
    mm.modeDevice = modeMemory->modeDevice;
    mm.size = bytes;
    mm.isOrigin = false;
    if (modeMemory->uvaPtr) {
      mm.uvaPtr = (modeMemory->uvaPtr + offset_);
    }

    return m;
  }

  void memory::copyFrom(const void *src,
                        const dim_t bytes,
                        const dim_t offset,
                        const occa::properties &props) {
    assertInitialized();

    udim_t bytes_ = ((bytes == -1) ? modeMemory->size : bytes);

    OCCA_ERROR("Trying to allocate negative bytes (" << bytes << ")",
               bytes >= -1);

    OCCA_ERROR("Cannot have a negative offset (" << offset << ")",
               offset >= 0);

    OCCA_ERROR("Destination memory has size [" << modeMemory->size << "],"
               << " trying to access [" << offset << ", " << (offset + bytes_) << "]",
               (bytes_ + offset) <= modeMemory->size);

    modeMemory->copyFrom(src, bytes_, offset, props);
  }

  void memory::copyFrom(const memory src,
                        const dim_t bytes,
                        const dim_t destOffset,
                        const dim_t srcOffset,
                        const occa::properties &props) {
    assertInitialized();

    udim_t bytes_ = ((bytes == -1) ? modeMemory->size : bytes);

    OCCA_ERROR("Trying to allocate negative bytes (" << bytes << ")",
               bytes >= -1);

    OCCA_ERROR("Cannot have a negative offset (" << destOffset << ")",
               destOffset >= 0);

    OCCA_ERROR("Cannot have a negative offset (" << srcOffset << ")",
               srcOffset >= 0);

    OCCA_ERROR("Source memory has size [" << src.modeMemory->size << "],"
               << " trying to access [" << srcOffset << ", " << (srcOffset + bytes_) << "]",
               (bytes_ + srcOffset) <= src.modeMemory->size);

    OCCA_ERROR("Destination memory has size [" << modeMemory->size << "],"
               << " trying to access [" << destOffset << ", " << (destOffset + bytes_) << "]",
               (bytes_ + destOffset) <= modeMemory->size);

    modeMemory->copyFrom(src.modeMemory, bytes_, destOffset, srcOffset, props);
  }

  void memory::copyTo(void *dest,
                      const dim_t bytes,
                      const dim_t offset,
                      const occa::properties &props) const {
    assertInitialized();

    udim_t bytes_ = ((bytes == -1) ? modeMemory->size : bytes);

    OCCA_ERROR("Trying to allocate negative bytes (" << bytes << ")",
               bytes >= -1);

    OCCA_ERROR("Cannot have a negative offset (" << offset << ")",
               offset >= 0);

    OCCA_ERROR("Source memory has size [" << modeMemory->size << "],"
               << " trying to access [" << offset << ", " << (offset + bytes_) << "]",
               (bytes_ + offset) <= modeMemory->size);

    modeMemory->copyTo(dest, bytes_, offset, props);
  }

  void memory::copyTo(memory dest,
                      const dim_t bytes,
                      const dim_t destOffset,
                      const dim_t srcOffset,
                      const occa::properties &props) const {
    assertInitialized();

    udim_t bytes_ = ((bytes == -1) ? modeMemory->size : bytes);

    OCCA_ERROR("Trying to allocate negative bytes (" << bytes << ")",
               bytes >= -1);

    OCCA_ERROR("Cannot have a negative offset (" << destOffset << ")",
               destOffset >= 0);

    OCCA_ERROR("Cannot have a negative offset (" << srcOffset << ")",
               srcOffset >= 0);

    OCCA_ERROR("Source memory has size [" << modeMemory->size << "],"
               << " trying to access [" << srcOffset << ", " << (srcOffset + bytes_) << "]",
               (bytes_ + srcOffset) <= modeMemory->size);

    OCCA_ERROR("Destination memory has size [" << dest.modeMemory->size << "],"
               << " trying to access [" << destOffset << ", " << (destOffset + bytes_) << "]",
               (bytes_ + destOffset) <= dest.modeMemory->size);

    dest.modeMemory->copyFrom(modeMemory, bytes_, destOffset, srcOffset, props);
  }

  void memory::copyFrom(const void *src,
                        const occa::properties &props) {
    copyFrom(src, -1, 0, props);
  }

  void memory::copyFrom(const memory src,
                        const occa::properties &props) {
    copyFrom(src, -1, 0, 0, props);
  }

  void memory::copyTo(void *dest,
                      const occa::properties &props) const {
    copyTo(dest, -1, 0, props);
  }

  void memory::copyTo(const memory dest,
                      const occa::properties &props) const {
    copyTo(dest, -1, 0, 0, props);
  }

  occa::memory memory::as(const dtype_t &dtype_) const {
    occa::memory mem = slice(0);
    mem.setDtype(dtype_);
    return mem;
  }

  occa::memory memory::clone() const {
    if (modeMemory) {
      return occa::device(modeMemory->modeDevice).malloc(size(),
                                                         *this,
                                                         properties());
    }
    return occa::memory();
  }

  void memory::free() {
    deleteRefs(true);
  }

  void memory::detach() {
    deleteRefs(false);
  }

  void memory::deleteRefs(const bool freeMemory) {
    if (modeMemory == NULL) {
      return;
    }

    modeDevice_t *modeDevice = modeMemory->modeDevice;

    // Free the actual backend memory object
    if (modeMemory->isOrigin) {
      modeDevice->bytesAllocated -= (modeMemory->size);

      if (modeMemory->uvaPtr) {
        void *ptr = modeMemory->ptr;
        void *uvaPtr = modeMemory->uvaPtr;

        uvaMap.erase(uvaPtr);
        modeDevice->uvaMap.erase(uvaPtr);

        // CPU case where memory is shared
        if (uvaPtr != ptr) {
          uvaMap.erase(ptr);
          modeDevice->uvaMap.erase(ptr);

          sys::free(uvaPtr);
        }
      }

      if (!freeMemory) {
        modeMemory->detach();
      }
    }

    // ~modeMemory_t NULLs all wrappers
    delete modeMemory;
    modeMemory = NULL;
  }

  memory null;
  //====================================

  std::ostream& operator << (std::ostream &out,
                             const occa::memory &memory) {
    out << memory.properties();
    return out;
  }

  namespace cpu {
    occa::memory wrapMemory(occa::device device,
                            void *ptr,
                            const udim_t bytes,
                            const occa::properties &props) {
      static occa::properties defaultProps(
        "use_host_pointer: true,"
        "own_host_pointer: false"
      );

      return device.malloc(bytes, ptr, defaultProps + props);
    }
  }
}
