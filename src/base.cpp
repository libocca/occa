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

#include "occa/base.hpp"
#include "occa/tools/sys.hpp"

namespace occa {

  //---[ Globals & Flags ]--------------
  properties settings;
  //====================================


  //---[ Registration ]-----------------
  strToModeMap_t& modeMap() {
    static strToModeMap_t modeMap_;
    return modeMap_;
  }

  bool modeIsEnabled(const std::string &mode) {
    return (modeMap().find(mode) != modeMap().end());
  }

  mode_v* getMode(const occa::properties &props) {
    if (!props.has("mode")) {
      std::cout << "No OCCA mode given, defaulting to [Serial] mode\n";
      return getMode("Serial");
    }
    return getMode(props["mode"]);
  }

  mode_v* getMode(const std::string &mode) {
    if (!modeIsEnabled(mode)) {
      std::cout << "OCCA mode [" << mode << "] is not enabled, defaulting to [Serial] mode\n";
      return modeMap()["Serial"];
    }
    return modeMap()[mode];
  }

  device_v* newModeDevice(const occa::properties &props) {
    return getMode(props)->newDevice(props);
  }

  kernel_v* newModeKernel(const occa::properties &props) {
    return getMode(props)->newKernel(props);
  }

  memory_v* newModeMemory(const occa::properties &props) {
    return getMode(props)->newMemory(props);
  }

  void freeModeDevice(device_v *dHandle) {
    delete dHandle;
  }

  void freeModeKernel(kernel_v *kHandle) {
    delete kHandle;
  }

  void freeModeMemory(memory_v *mHandle) {
    delete mHandle;
  }

  std::string& mode_v::name() {
    return modeName;
  }
  //====================================


  //---[ Memory ]-----------------------
  void memcpy(void *dest, const void *src,
              const udim_t bytes,
              const properties &props) {

    ptrRangeMap_t::iterator srcIt  = uvaMap.find(const_cast<void*>(src));
    ptrRangeMap_t::iterator destIt = uvaMap.find(dest);

    occa::memory_v *srcMem  = ((srcIt != uvaMap.end())  ? (srcIt->second)  : NULL);
    occa::memory_v *destMem = ((destIt != uvaMap.end()) ? (destIt->second) : NULL);

    const udim_t srcOff  = (srcMem  ? (((char*) src)  - ((char*) srcMem->uvaPtr))  : 0);
    const udim_t destOff = (destMem ? (((char*) dest) - ((char*) destMem->uvaPtr)) : 0);

    const bool usingSrcPtr  = ((srcMem  == NULL) || srcMem->isManaged());
    const bool usingDestPtr = ((destMem == NULL) || destMem->isManaged());

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
              const udim_t bytes,
              const udim_t offset,
              const properties &props) {

    dest.copyFrom(src, bytes, offset, props);
  }

  void memcpy(void *dest, memory src,
              const udim_t bytes,
              const udim_t offset,
              const properties &props) {

    src.copyTo(dest, bytes, offset, props);
  }

  void memcpy(memory dest, memory src,
              const udim_t bytes,
              const udim_t destOffset,
              const udim_t srcOffset,
              const properties &props) {

    dest.copyFrom(src, bytes, destOffset, srcOffset, props);
  }
  //====================================


  //---[ Device Functions ]-------------
  device currentDevice;

  device getCurrentDevice() {
    if (currentDevice.getDHandle() == NULL) {
      currentDevice = host();
    }
    return currentDevice;
  }

  device host() {
    static device _host;
    if (_host.getDHandle() == NULL) {
      _host = occa::device(newModeDevice(occa::properties("mode = Serial")));
    }
    return _host;
  }

  void setDevice(device d) {
    currentDevice = d;
  }

  void setDevice(const std::string &props) {
    currentDevice = device(props);
  }

  void setDevice(const properties &props) {
    currentDevice = device(props);
  }

  std::vector<device>& getDeviceList() {
    static std::vector<device> deviceList;
    static mutex_t mutex;

    mutex.lock();
    if (deviceList.size() == 0) {
      strToModeMapIterator it = modeMap().begin();
      while (it != modeMap().end()) {
        device_v* dHandle = it->second->newDevice();
        dHandle->appendAvailableDevices(deviceList);
        freeModeDevice(dHandle);
        ++it;
      }
    }
    mutex.unlock();

    return deviceList;
  }

  properties& deviceProperties() {
    return getCurrentDevice().properties();
  }

  void flush() {
    getCurrentDevice().flush();
  }

  void finish() {
    getCurrentDevice().finish();
  }

  void waitFor(streamTag tag) {
    getCurrentDevice().waitFor(tag);
  }

  stream createStream() {
    return getCurrentDevice().createStream();
  }

  stream getStream() {
    return getCurrentDevice().getStream();
  }

  void setStream(stream s) {
    getCurrentDevice().setStream(s);
  }

  stream wrapStream(void *handle_) {
    return getCurrentDevice().wrapStream(handle_);
  }

  streamTag tagStream() {
    return getCurrentDevice().tagStream();
  }

  //---[ Kernel Functions ]-------------
  kernel buildKernel(const std::string &str,
                     const std::string &functionName,
                     const properties &props) {

    return getCurrentDevice().buildKernel(str,
                                          functionName,
                                          props);
  }

  kernel buildKernelFromString(const std::string &content,
                               const std::string &functionName,
                               const properties &props) {

    return getCurrentDevice().buildKernelFromString(content, functionName, props);
  }

  kernel buildKernelFromSource(const std::string &filename,
                               const std::string &functionName,
                               const properties &props) {

    return getCurrentDevice().buildKernelFromSource(filename,
                                                    functionName,
                                                    props);
  }

  kernel buildKernelFromBinary(const std::string &filename,
                               const std::string &functionName) {

    return getCurrentDevice().buildKernelFromBinary(filename,
                                                    functionName);
  }

  //---[ Memory Functions ]-------------
  occa::memory malloc(const udim_t bytes,
                      void *src,
                      const properties &props) {

    return getCurrentDevice().malloc(bytes, src, props);
  }

  void* managedAlloc(const udim_t bytes,
                     void *src,
                     const properties &props) {

    return getCurrentDevice().managedAlloc(bytes, src, props);
  }

  occa::memory wrapMemory(void *handle_,
                          const udim_t bytes,
                          const occa::properties &props) {

    return getCurrentDevice().wrapMemory(handle_, bytes, props);
  }
  //====================================

  //---[ Free Functions ]---------------
  void free(device d) {
    d.free();
  }

  void free(stream s) {
    getCurrentDevice().freeStream(s);
  }

  void free(kernel k) {
    k.free();
  }

  void free(memory m) {
    m.free();
  }
  //====================================

  // [REFORMAT]
  void printAvailableDevices() {
  }
  //====================================
}
