/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2017 David Medina and Tim Warburton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */

#include <map>

#include "occa/memory.hpp"
#include "occa/device.hpp"
#include "occa/uva.hpp"
#include "occa/tools/sys.hpp"

namespace occa {
  //---[ memory_v ]---------------------
  memory_v::memory_v(const occa::properties &properties_) {
    memInfo = uvaFlag::none;
    properties = properties_;

    handle = NULL;
    uvaPtr = NULL;

    dHandle = NULL;
    size    = 0;
  }

  memory_v::~memory_v() {}

  void memory_v::initFrom(const memory_v &m) {
    memInfo = m.memInfo;
    properties = m.properties;

    handle = m.handle;
    uvaPtr = m.uvaPtr;

    dHandle = m.dHandle;
    size    = m.size;
  }

  bool memory_v::isManaged() const {
    return (memInfo & uvaFlag::isManaged);
  }

  bool memory_v::inDevice() const {
    return (memInfo & uvaFlag::inDevice);
  }

  bool memory_v::isStale() const {
    return (memInfo & uvaFlag::isStale);
  }

  void* memory_v::uvaHandle() {
    return handle;
  }

  //---[ memory ]-----------------------
  memory::memory() :
    mHandle(NULL) {}

  memory::memory(void *uvaPtr) :
    mHandle(NULL) {
    ptrRangeMap_t::iterator it = uvaMap.find(uvaPtr);
    if (it != uvaMap.end()) {
      setMHandle(it->second);
    } else {
      setMHandle((memory_v*) uvaPtr);
    }
  }

  memory::memory(memory_v *mHandle_) :
    mHandle(NULL) {
    setMHandle(mHandle_);
  }

  memory::memory(const memory &m) :
    mHandle(NULL) {
    setMHandle(m.mHandle);
  }

  memory& memory::operator = (const memory &m) {
    setMHandle(m.mHandle);
    return *this;
  }

  memory::~memory() {
    removeMHandleRef();
  }

  void memory::setMHandle(memory_v *mHandle_) {
    if (mHandle != mHandle_) {
      removeMHandleRef();
      mHandle = mHandle_;
      mHandle->addRef();
    }
  }

  void memory::setDHandle(device_v *dHandle) {
    mHandle->dHandle = dHandle;
    // If this is the very first reference, update the device references
    if (mHandle->getRefs() == 1) {
      mHandle->dHandle->addRef();
    }
  }

  void memory::removeMHandleRef() {
    if (mHandle && !mHandle->removeRef()) {
      free();
      device::removeDHandleRefFrom(mHandle->dHandle);
      delete mHandle;
      mHandle = NULL;
    }
  }

  void memory::dontUseRefs() {
    if (mHandle) {
      mHandle->dontUseRefs();
    }
  }

  bool memory::isInitialized() const {
    return (mHandle != NULL);
  }

  memory& memory::swap(memory &m) {
    memory_v *mHandle_ = mHandle;
    mHandle   = m.mHandle;
    m.mHandle = mHandle_;
    return *this;
  }

  memory_v* memory::getMHandle() const {
    return mHandle;
  }

  device_v* memory::getDHandle() const {
    return mHandle->dHandle;
  }

  occa::device memory::getDevice() const {
    return occa::device(mHandle->dHandle);
  }

  memory::operator kernelArg() const {
    kernelArg kArg = mHandle->makeKernelArg();
    kArg.args[0].mHandle = mHandle;
    kArg.args[0].dHandle = mHandle->dHandle;
    return kArg;
  }

  const std::string& memory::mode() const {
    return device(mHandle->dHandle).mode();
  }

  udim_t memory::size() const {
    if (mHandle == NULL) {
      return 0;
    }
    return mHandle->size;
  }

  bool memory::isManaged() const {
    return mHandle->isManaged();
  }

  bool memory::inDevice() const {
    return mHandle->inDevice();
  }

  bool memory::isStale() const {
    return mHandle->isStale();
  }

  void* memory::getHandle(const occa::properties &props) const {
    return mHandle->getHandle(props);
  }

  void memory::setupUva() {
    if ( !(mHandle->dHandle->hasSeparateMemorySpace()) ) {
      mHandle->uvaPtr = mHandle->uvaHandle();
    } else {
      mHandle->uvaPtr = sys::malloc(mHandle->size);
    }

    ptrRange_t uvaRange;
    uvaRange.start = (char*) (mHandle->uvaPtr);
    uvaRange.end   = (uvaRange.start + mHandle->size);

    uvaMap[uvaRange]                   = mHandle;
    mHandle->dHandle->uvaMap[uvaRange] = mHandle;

    // Needed for kernelArg.void_ -> mHandle checks
    if (mHandle->uvaPtr != mHandle->handle) {
      uvaMap[mHandle->handle] = mHandle;
    }
  }

  void memory::startManaging() {
    mHandle->memInfo |= uvaFlag::isManaged;
  }

  void memory::stopManaging() {
    mHandle->memInfo &= ~uvaFlag::isManaged;
  }

  void memory::syncToDevice(const dim_t bytes,
                            const dim_t offset) {
    udim_t bytes_ = ((bytes == -1) ? mHandle->size : bytes);

    OCCA_ERROR("Trying to copy negative bytes (" << bytes << ")",
               bytes >= -1);
    OCCA_ERROR("Cannot have a negative offset (" << offset << ")",
               offset >= 0);

    if (bytes_ == 0) {
      return;
    }

    OCCA_ERROR("Memory has size [" << mHandle->size << "],"
               << " trying to access [ " << offset << " , " << (offset + bytes_) << " ]",
               (bytes_ + offset) <= mHandle->size);

    if (!mHandle->dHandle->hasSeparateMemorySpace()) {
      return;
    }

    copyFrom(mHandle->uvaPtr, bytes_, offset);

    mHandle->memInfo |=  uvaFlag::inDevice;
    mHandle->memInfo &= ~uvaFlag::isStale;

    removeFromStaleMap(mHandle);
  }

  void memory::syncFromDevice(const dim_t bytes,
                              const dim_t offset) {
    udim_t bytes_ = ((bytes == -1) ? mHandle->size : bytes);

    OCCA_ERROR("Trying to copy negative bytes (" << bytes << ")",
               bytes >= -1);
    OCCA_ERROR("Cannot have a negative offset (" << offset << ")",
               offset >= 0);

    if (bytes_ == 0) {
      return;
    }

    OCCA_ERROR("Memory has size [" << mHandle->size << "],"
               << " trying to access [ " << offset << " , " << (offset + bytes_) << " ]",
               (bytes_ + offset) <= mHandle->size);

    if (!mHandle->dHandle->hasSeparateMemorySpace()) {
      return;
    }

    copyTo(mHandle->uvaPtr, bytes_, offset);

    mHandle->memInfo &= ~uvaFlag::inDevice;
    mHandle->memInfo &= ~uvaFlag::isStale;

    removeFromStaleMap(mHandle);
  }

  bool memory::uvaIsStale() const {
    return (mHandle && mHandle->isStale());
  }

  void memory::uvaMarkStale() {
    if (mHandle != NULL) {
      mHandle->memInfo |= uvaFlag::isStale;
    }
  }

  void memory::uvaMarkFresh() {
    if (mHandle != NULL) {
      mHandle->memInfo &= ~uvaFlag::isStale;
    }
  }

  occa::memory memory::operator + (const dim_t offset) const {
    OCCA_ERROR("Cannot have a negative offset (" << offset << ")",
               offset >= 0);
    OCCA_ERROR("Cannot have an offset greater than the memory size ("
               << offset << " >= " << size() << ")",
               offset < (dim_t) size());

    bool needsFree;
    occa::memory m(mHandle->addOffset(offset, needsFree));
    m.mHandle->dHandle = mHandle->dHandle;
    m.mHandle->size = size() - offset;
    if (!needsFree) {
      m.dontUseRefs();
    }
    return m;
  }

  occa::memory& memory::operator += (const dim_t offset) {
    *this = (*this + offset);
    return *this;
  }

  void memory::copyFrom(const void *src,
                        const dim_t bytes,
                        const dim_t offset,
                        const occa::properties &props) {
    udim_t bytes_ = ((bytes == -1) ? mHandle->size : bytes);

    OCCA_ERROR("Trying to allocate negative bytes (" << bytes << ")",
               bytes >= -1);
    OCCA_ERROR("Cannot have a negative offset (" << offset << ")",
               offset >= 0);
    OCCA_ERROR("Destination memory has size [" << mHandle->size << "],"
               << "trying to access [ " << offset << " , " << (offset + bytes_) << " ]",
               (bytes_ + offset) <= mHandle->size);

    mHandle->copyFrom(src, bytes_, offset, props);
  }

  void memory::copyFrom(const memory src,
                        const dim_t bytes,
                        const dim_t destOffset,
                        const dim_t srcOffset,
                        const occa::properties &props) {
    udim_t bytes_ = ((bytes == -1) ? mHandle->size : bytes);

    OCCA_ERROR("Trying to allocate negative bytes (" << bytes << ")",
               bytes >= -1);
    OCCA_ERROR("Cannot have a negative offset (" << destOffset << ")",
               destOffset >= 0);
    OCCA_ERROR("Cannot have a negative offset (" << srcOffset << ")",
               srcOffset >= 0);
    OCCA_ERROR("Source memory has size [" << src.mHandle->size << "],"
               << "trying to access [ " << srcOffset << " , " << (srcOffset + bytes_) << " ]",
               (bytes_ + srcOffset) <= src.mHandle->size);
    OCCA_ERROR("Destination memory has size [" << mHandle->size << "],"
               << "trying to access [ " << destOffset << " , " << (destOffset + bytes_) << " ]",
               (bytes_ + destOffset) <= mHandle->size);

    mHandle->copyFrom(src.mHandle, bytes_, destOffset, srcOffset, props);
  }

  void memory::copyTo(void *dest,
                      const dim_t bytes,
                      const dim_t offset,
                      const occa::properties &props) const {
    udim_t bytes_ = ((bytes == -1) ? mHandle->size : bytes);

    OCCA_ERROR("Trying to allocate negative bytes (" << bytes << ")",
               bytes >= -1);
    OCCA_ERROR("Cannot have a negative offset (" << offset << ")",
               offset >= 0);
    OCCA_ERROR("Source memory has size [" << mHandle->size << "],"
               << "trying to access [ " << offset << " , " << (offset + bytes_) << " ]",
               (bytes_ + offset) <= mHandle->size);

    mHandle->copyTo(dest, bytes_, offset, props);
  }

  void memory::copyTo(memory dest,
                      const dim_t bytes,
                      const dim_t destOffset,
                      const dim_t srcOffset,
                      const occa::properties &props) const {
    udim_t bytes_ = ((bytes == -1) ? mHandle->size : bytes);

    OCCA_ERROR("Trying to allocate negative bytes (" << bytes << ")",
               bytes >= -1);
    OCCA_ERROR("Cannot have a negative offset (" << destOffset << ")",
               destOffset >= 0);
    OCCA_ERROR("Cannot have a negative offset (" << srcOffset << ")",
               srcOffset >= 0);
    OCCA_ERROR("Source memory has size [" << mHandle->size << "],"
               << "trying to access [ " << srcOffset << " , " << (srcOffset + bytes_) << " ]",
               (bytes_ + srcOffset) <= mHandle->size);
    OCCA_ERROR("Destination memory has size [" << dest.mHandle->size << "],"
               << "trying to access [ " << destOffset << " , " << (destOffset + bytes_) << " ]",
               (bytes_ + destOffset) <= dest.mHandle->size);

    dest.mHandle->copyFrom(mHandle, bytes_, destOffset, srcOffset, props);
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

  void memory::free() {
    deleteRefs(true);
  }

  void memory::detach() {
    deleteRefs(false);
  }

  void memory::deleteRefs(const bool freeMemory) {
    if (mHandle == NULL) {
      return;
    }
    mHandle->dHandle->bytesAllocated -= (mHandle->size);

    if (mHandle->uvaPtr) {
      uvaMap.erase(mHandle->uvaPtr);
      mHandle->dHandle->uvaMap.erase(mHandle->uvaPtr);

      // CPU case where memory is shared
      if (mHandle->uvaPtr != mHandle->handle) {
        uvaMap.erase(mHandle->handle);
        mHandle->dHandle->uvaMap.erase(mHandle->uvaPtr);

        ::free(mHandle->uvaPtr);
        mHandle->uvaPtr = NULL;
      }
    }

    if (freeMemory) {
      mHandle->free();
    } else {
      mHandle->detach();
    }
  }
}
