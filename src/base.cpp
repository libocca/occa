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

#include "occa/base.hpp"
#include "occa/mode.hpp"
#include "occa/par/tls.hpp"
#include "occa/tools/env.hpp"
#include "occa/tools/sys.hpp"

namespace occa {
  //---[ Globals & Flags ]--------------
  properties& settings() {
    static tls<properties> settings_;
    properties& props = settings_.value();
    if (!props.isInitialized()) {
      props = env::baseSettings();
    }
    return props;
  }
  //====================================

  //---[ Device Functions ]-------------
  device host() {
    static device dev;
    if (!dev.isInitialized()) {
      dev = occa::device(newModeDevice("mode: 'Serial'"));
    }
    return dev;
  }

  device& getDevice() {
    static tls<device> tdev;
    device &dev = tdev.value();
    if (!dev.isInitialized()) {
      dev = host();
    }
    return dev;
  }

  void setDevice(device d) {
    getDevice() = d;
  }

  void setDevice(const occa::properties &props) {
    getDevice() = device(props);
  }

  const occa::properties& deviceProperties() {
    return getDevice().properties();
  }

  void loadKernels(const std::string &library) {
    getDevice().loadKernels(library);
  }

  void finish() {
    getDevice().finish();
  }

  void waitFor(streamTag tag) {
    getDevice().waitFor(tag);
  }

  stream createStream() {
    return getDevice().createStream();
  }

  stream getStream() {
    return getDevice().getStream();
  }

  void setStream(stream s) {
    getDevice().setStream(s);
  }

  stream wrapStream(void *handle_, const occa::properties &props) {
    return getDevice().wrapStream(handle_, props);
  }

  streamTag tagStream() {
    return getDevice().tagStream();
  }

  //---[ Kernel Functions ]-------------
  kernel buildKernel(const std::string &filename,
                     const std::string &kernelName,
                     const occa::properties &props) {

    return getDevice().buildKernel(filename,
                                   kernelName,
                                   props);
  }

  kernel buildKernelFromString(const std::string &content,
                               const std::string &kernelName,
                               const occa::properties &props) {

    return getDevice().buildKernelFromString(content, kernelName, props);
  }

  kernel buildKernelFromBinary(const std::string &filename,
                               const std::string &kernelName,
                               const occa::properties &props) {

    return getDevice().buildKernelFromBinary(filename, kernelName, props);
  }

  //---[ Memory Functions ]-------------
  occa::memory malloc(const dim_t bytes,
                      const void *src,
                      const occa::properties &props) {

    return getDevice().malloc(bytes, src, props);
  }

  void* umalloc(const dim_t bytes,
                const void *src,
                const occa::properties &props) {

    return getDevice().umalloc(bytes, src, props);
  }

  void memcpy(void *dest, const void *src,
              const dim_t bytes,
              const occa::properties &props) {

    ptrRangeMap::iterator srcIt  = uvaMap.find(const_cast<void*>(src));
    ptrRangeMap::iterator destIt = uvaMap.find(dest);

    occa::memory_v *srcMem  = ((srcIt  != uvaMap.end()) ? (srcIt->second)  : NULL);
    occa::memory_v *destMem = ((destIt != uvaMap.end()) ? (destIt->second) : NULL);

    const udim_t srcOff  = (srcMem  ? (((char*) src)  - srcMem->uvaPtr)  : 0);
    const udim_t destOff = (destMem ? (((char*) dest) - destMem->uvaPtr) : 0);

    const bool usingSrcPtr  = ((srcMem  == NULL) ||
                               ((srcMem->isManaged() && !srcMem->inDevice())));
    const bool usingDestPtr = ((destMem  == NULL) ||
                               ((destMem->isManaged() && !destMem->inDevice())));

    if (usingSrcPtr && usingDestPtr) {
      ::memcpy(dest, src, bytes);
    } else if (usingSrcPtr) {
      destMem->copyFrom(src, bytes, destOff, props);
    } else if (usingDestPtr) {
      srcMem->copyTo(dest, bytes, srcOff, props);
    } else {
      // Auto-detects peer-to-peer stuff
      occa::memory srcMemory(srcMem);
      occa::memory destMemory(destMem);
      destMemory.copyFrom(srcMemory, bytes, destOff, srcOff, props);
    }
  }

  void memcpy(memory dest, const void *src,
              const dim_t bytes,
              const dim_t offset,
              const occa::properties &props) {

    dest.copyFrom(src, bytes, offset, props);
  }

  void memcpy(void *dest, memory src,
              const dim_t bytes,
              const dim_t offset,
              const occa::properties &props) {

    src.copyTo(dest, bytes, offset, props);
  }

  void memcpy(memory dest, memory src,
              const dim_t bytes,
              const dim_t destOffset,
              const dim_t srcOffset,
              const occa::properties &props) {

    dest.copyFrom(src, bytes, destOffset, srcOffset, props);
  }

  void memcpy(void *dest, const void *src,
              const occa::properties &props) {
    memcpy(dest, src, -1, props);
  }

  void memcpy(memory dest, const void *src,
              const occa::properties &props) {
    memcpy(dest, src, -1, 0, props);
  }

  void memcpy(void *dest, memory src,
              const occa::properties &props) {
    memcpy(dest, src, -1, 0, props);
  }

  void memcpy(memory dest, memory src,
              const occa::properties &props) {
    memcpy(dest, src, -1, 0, 0, props);
  }
  //====================================

  //---[ Free Functions ]---------------
  void free(device d) {
    d.free();
  }

  void free(stream s) {
    getDevice().freeStream(s);
  }

  void free(kernel k) {
    k.free();
  }

  void free(memory m) {
    m.free();
  }
  //====================================

  void printModeInfo() {
    strToModeMap &modes = modeMap();
    strToModeMapIterator it = modes.begin();
    styling::table table;
    int serialIdx = 0;
    int idx = 0;
    while (it != modes.end()) {
      if (it->first == "Serial") {
        serialIdx = idx;
      }
      table.add(it->second->getDescription());
      ++it;
      ++idx;
    }
    styling::section serialSection = table.sections[serialIdx];
    table.sections[serialIdx] = table.sections[0];
    table.sections[0] = serialSection;
    std::cout << table.toString();
  }
}
