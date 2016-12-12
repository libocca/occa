/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2016 David Medina and Tim Warburton
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

  bool memory_v::leftInDevice() const {
    return (memInfo & uvaFlag::leftInDevice);
  }

  bool memory_v::isDirty() const {
    return (memInfo & uvaFlag::isDirty);
  }

  void* memory_v::uvaHandle() {
    return handle;
  }

  //---[ memory ]-----------------------
  memory::memory() :
    mHandle(NULL) {}

  memory::memory(void *uvaPtr) {
    // Default to uvaPtr is actually a memory_v*
    memory_v *mHandle_ = (memory_v*) uvaPtr;
    ptrRangeMap_t::iterator it = uvaMap.find(uvaPtr);

    if (it != uvaMap.end()) {
      mHandle_ = it->second;
    }

    mHandle = mHandle_;
  }

  memory::memory(memory_v *mHandle_) :
    mHandle(mHandle_) {}

  memory::memory(const memory &m) :
    mHandle(m.mHandle) {}

  memory& memory::swap(memory &m) {
    memory_v *tmp = mHandle;
    mHandle       = m.mHandle;
    m.mHandle     = tmp;

    return *this;
  }

  memory& memory::operator = (const memory &m) {
    mHandle = m.mHandle;
    return *this;
  }

  void memory::checkIfInitialized() const {
    OCCA_CHECK(mHandle != NULL,
               "Memory is not initialized");
  }

  memory_v* memory::getMHandle() {
    checkIfInitialized();
    return mHandle;
  }

  device_v* memory::getDHandle() {
    checkIfInitialized();
    return mHandle->dHandle;
  }

  memory::operator kernelArg() const {
    kernelArg kArg = mHandle->makeKernelArg();
    kArg.arg.mHandle = mHandle;
    kArg.arg.dHandle = mHandle->dHandle;
    return kArg;
  }

  const std::string& memory::mode() {
    checkIfInitialized();
    return device(mHandle->dHandle).mode();
  }

  udim_t memory::bytes() const {
    if (mHandle == NULL) {
      return 0;
    }
    return mHandle->size;
  }

  bool memory::isManaged() const {
    return (mHandle->memInfo & uvaFlag::isManaged);
  }

  bool memory::inDevice() const {
    return (mHandle->memInfo & uvaFlag::inDevice);
  }

  bool memory::leftInDevice() const {
    return (mHandle->memInfo & uvaFlag::leftInDevice);
  }

  bool memory::isDirty() const {
    return (mHandle->memInfo & uvaFlag::isDirty);
  }

  void* memory::getHandle(const occa::properties &props) {
    checkIfInitialized();
    return mHandle->getHandle(props);
  }

  void memory::manage() {
    checkIfInitialized();

    if ( !(mHandle->dHandle->fakesUva()) ) {
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

    mHandle->memInfo |= uvaFlag::isManaged;
  }

  void memory::syncToDevice(const dim_t bytes,
                            const dim_t offset) {
    checkIfInitialized();

    udim_t bytes_ = ((bytes == -1) ? mHandle->size : bytes);

    OCCA_CHECK(bytes >= -1,
               "Trying to allocate negative bytes (" << bytes << ")");
    OCCA_CHECK(offset >= 0,
               "Cannot have a negative offset (" << offset << ")");

    if (mHandle->dHandle->fakesUva()) {
      OCCA_CHECK((bytes_ + offset) <= mHandle->size,
                 "Memory has size [" << mHandle->size << "],"
                 << "trying to access [ " << offset << " , " << (offset + bytes_) << " ]");

      copyTo(mHandle->uvaPtr, bytes_, offset);

      mHandle->memInfo |=  uvaFlag::inDevice;
      mHandle->memInfo &= ~uvaFlag::isDirty;

      removeFromDirtyMap(mHandle);
    }
  }

  void memory::syncFromDevice(const dim_t bytes,
                              const dim_t offset) {
    checkIfInitialized();

    udim_t bytes_ = ((bytes == 0) ? mHandle->size : bytes);

    OCCA_CHECK(bytes >= -1,
               "Trying to allocate negative bytes (" << bytes << ")");
    OCCA_CHECK(offset >= 0,
               "Cannot have a negative offset (" << offset << ")");

    if (mHandle->dHandle->fakesUva()) {
      OCCA_CHECK((bytes_ + offset) <= mHandle->size,
                 "Memory has size [" << mHandle->size << "],"
                 << "trying to access [ " << offset << " , " << (offset + bytes_) << " ]");

      copyFrom(mHandle->uvaPtr, bytes_, offset);

      mHandle->memInfo &= ~uvaFlag::inDevice;
      mHandle->memInfo &= ~uvaFlag::isDirty;

      removeFromDirtyMap(mHandle);
    }
  }

  bool memory::uvaIsDirty() {
    checkIfInitialized();
    return (mHandle && mHandle->isDirty());
  }

  void memory::uvaMarkDirty() {
    checkIfInitialized();
    if (mHandle != NULL) {
      mHandle->memInfo |= uvaFlag::isDirty;
    }
  }

  void memory::uvaMarkClean() {
    checkIfInitialized();
    if (mHandle != NULL) {
      mHandle->memInfo &= ~uvaFlag::isDirty;
    }
  }

  void memory::copyFrom(const void *src,
                        const dim_t bytes,
                        const dim_t offset,
                        const occa::properties &props) {
    checkIfInitialized();

    udim_t bytes_ = ((bytes == -1) ? mHandle->size : bytes);

    OCCA_CHECK(bytes >= -1,
               "Trying to allocate negative bytes (" << bytes << ")");
    OCCA_CHECK(offset >= 0,
               "Cannot have a negative offset (" << offset << ")");
    OCCA_CHECK((bytes_ + offset) <= mHandle->size,
               "Destination memory has size [" << mHandle->size << "],"
               << "trying to access [ " << offset << " , " << (offset + bytes_) << " ]");

    mHandle->copyFrom(src, bytes_, offset, props);
  }

  void memory::copyFrom(const memory src,
                        const dim_t bytes,
                        const dim_t destOffset,
                        const dim_t srcOffset,
                        const occa::properties &props) {
    checkIfInitialized();

    udim_t bytes_ = ((bytes == -1) ? mHandle->size : bytes);

    OCCA_CHECK(bytes >= -1,
               "Trying to allocate negative bytes (" << bytes << ")");
    OCCA_CHECK(destOffset >= 0,
               "Cannot have a negative offset (" << destOffset << ")");
    OCCA_CHECK(srcOffset >= 0,
               "Cannot have a negative offset (" << srcOffset << ")");
    OCCA_CHECK((bytes_ + srcOffset) <= src.mHandle->size,
               "Source memory has size [" << src.mHandle->size << "],"
               << "trying to access [ " << srcOffset << " , " << (srcOffset + bytes_) << " ]");
    OCCA_CHECK((bytes_ + destOffset) <= mHandle->size,
               "Destination memory has size [" << mHandle->size << "],"
               << "trying to access [ " << destOffset << " , " << (destOffset + bytes_) << " ]");

    mHandle->copyFrom(src.mHandle, bytes_, destOffset, srcOffset, props);
  }

  void memory::copyTo(void *dest,
                      const dim_t bytes,
                      const dim_t offset,
                      const occa::properties &props) {
    checkIfInitialized();

    udim_t bytes_ = ((bytes == -1) ? mHandle->size : bytes);

    OCCA_CHECK(bytes >= -1,
               "Trying to allocate negative bytes (" << bytes << ")");
    OCCA_CHECK(offset >= 0,
               "Cannot have a negative offset (" << offset << ")");
    OCCA_CHECK((bytes_ + offset) <= mHandle->size,
               "Source memory has size [" << mHandle->size << "],"
               << "trying to access [ " << offset << " , " << (offset + bytes_) << " ]");

    mHandle->copyTo(dest, bytes_, offset, props);
  }

  void memory::copyTo(memory dest,
                      const dim_t bytes,
                      const dim_t destOffset,
                      const dim_t srcOffset,
                      const occa::properties &props) {
    checkIfInitialized();

    udim_t bytes_ = ((bytes == -1) ? mHandle->size : bytes);

    OCCA_CHECK(bytes >= -1,
               "Trying to allocate negative bytes (" << bytes << ")");
    OCCA_CHECK(destOffset >= 0,
               "Cannot have a negative offset (" << destOffset << ")");
    OCCA_CHECK(srcOffset >= 0,
               "Cannot have a negative offset (" << srcOffset << ")");
    OCCA_CHECK((bytes_ + srcOffset) <= mHandle->size,
               "Source memory has size [" << mHandle->size << "],"
               << "trying to access [ " << srcOffset << " , " << (srcOffset + bytes_) << " ]");
    OCCA_CHECK((bytes_ + destOffset) <= dest.mHandle->size,
               "Destination memory has size [" << dest.mHandle->size << "],"
               << "trying to access [ " << destOffset << " , " << (destOffset + bytes_) << " ]");

    dest.mHandle->copyFrom(mHandle, bytes_, destOffset, srcOffset, props);
  }

  void memory::free() {
    deleteRefs(true);
  }

  void memory::detach() {
    deleteRefs(false);
  }

  void memory::deleteRefs(const bool free) {
    checkIfInitialized();

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

    if (free) {
      mHandle->free();
    } else {
      mHandle->detach();
    }

    delete mHandle;
    mHandle = NULL;
  }
}
