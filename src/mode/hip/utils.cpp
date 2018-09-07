/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 David Medina and Tim Warburton
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

#include <occa/defines.hpp>

#if OCCA_HIP_ENABLED

#include <occa/core/device.hpp>
#include <occa/core/memory.hpp>
#include <occa/tools/string.hpp>
#include <occa/tools/sys.hpp>
#include <occa/mode/hip/device.hpp>
#include <occa/mode/hip/memory.hpp>
#include <occa/mode/hip/utils.hpp>

namespace occa {
  //---[ Helper Functions ]-----------
  namespace hip {
    void init() {
      static bool isInitialized = false;
      if (!isInitialized) {
        hipInit(0);
        isInitialized = true;
      }
    }

    int getDeviceCount() {
      int deviceCount;
      OCCA_HIP_ERROR("Finding Number of Devices",
                     hipGetDeviceCount(&deviceCount));
      return deviceCount;
    }

    hipDevice_t getDevice(const int id) {
      hipDevice_t device;
      OCCA_HIP_ERROR("Getting hipDevice_t",
                     hipDeviceGet(&device, id));
      return device;
    }

    udim_t getDeviceMemorySize(hipDevice_t device) {
      size_t bytes;
      OCCA_HIP_ERROR("Finding available memory on device",
                     hipDeviceTotalMem(&bytes, device));
      return bytes;
    }

    std::string getVersion() {
      std::stringstream ss;
      int driverVersion;
      OCCA_HIP_ERROR("Finding HIP driver version",
                     hipDriverGetVersion(&driverVersion));
      ss << driverVersion;

      return ss.str();
    }

    void enablePeerToPeer(hipCtx_t context) {

      OCCA_HIP_ERROR("Enabling Peer-to-Peer",
                     hipCtxEnablePeerAccess(context, 0) );
    }

    void checkPeerToPeer(hipDevice_t destDevice,
                         hipDevice_t srcDevice) {
      int canAccessPeer;

      OCCA_HIP_ERROR("Checking Peer-to-Peer Connection",
                     hipDeviceCanAccessPeer(&canAccessPeer,
                                            destDevice,
                                            srcDevice));

      OCCA_ERROR("Checking Peer-to-Peer Connection",
                 (canAccessPeer == 1));
    }

    void peerToPeerMemcpy(hipDevice_t destDevice,
                          hipCtx_t destContext,
                          hipDeviceptr_t destMemory,
                          hipDevice_t srcDevice,
                          hipCtx_t srcContext,
                          hipDeviceptr_t srcMemory,
                          const udim_t bytes,
                          hipStream_t usingStream) {

      peerToPeerMemcpy(destDevice, destContext, destMemory,
                       srcDevice , srcContext , srcMemory ,
                       bytes,
                       usingStream, false);
    }


    void asyncPeerToPeerMemcpy(hipDevice_t destDevice,
                               hipCtx_t destContext,
                               hipDeviceptr_t destMemory,
                               hipDevice_t srcDevice,
                               hipCtx_t srcContext,
                               hipDeviceptr_t srcMemory,
                               const udim_t bytes,
                               hipStream_t usingStream) {

      peerToPeerMemcpy(destDevice, destContext, destMemory,
                       srcDevice , srcContext , srcMemory ,
                       bytes,
                       usingStream, true);
    }

    void peerToPeerMemcpy(hipDevice_t destDevice,
                          hipCtx_t destContext,
                          hipDeviceptr_t destMemory,
                          hipDevice_t srcDevice,
                          hipCtx_t srcContext,
                          hipDeviceptr_t srcMemory,
                          const udim_t bytes,
                          hipStream_t usingStream,
                          const bool isAsync) {

      OCCA_FORCE_ERROR("HIP version ["
                       << hip::getVersion()
                       << "] does not support Peer-to-Peer");

    }

    void advise(occa::memory mem, advice_t advice, const dim_t bytes) {
      advise(mem, advice, bytes, mem.getDevice());
    }

    void advise(occa::memory mem, advice_t advice, occa::device device) {
      advise(mem, advice, -1, device);
    }

    void advise(occa::memory mem, advice_t advice, const dim_t bytes, occa::device device) {

      OCCA_FORCE_ERROR("HIP version ["
                       << hip::getVersion()
                       << "] does not support unified memory advising");

    }

    void prefetch(occa::memory mem, const dim_t bytes) {
      prefetch(mem, bytes, mem.getDevice());
    }

    void prefetch(occa::memory mem, occa::device device) {
      prefetch(mem, -1, device);
    }

    void prefetch(occa::memory mem, const dim_t bytes, occa::device device) {
      OCCA_ERROR("Memory allocated with mode [" << mem.mode() << "], not [HIP]",
                 mem.mode() == "HIP");

      OCCA_FORCE_ERROR("HIP version ["
                       << hip::getVersion()
                       << "] does not support unified memory prefetching");
    }

    hipCtx_t getContext(occa::device device) {
      return ((hip::device*) device.getModeDevice())->hipContext;
    }

    void* getMappedPtr(occa::memory mem) {
      hip::memory *handle = (hip::memory*) mem.getMHandle();
      return handle ? handle->mappedPtr : NULL;
    }

    occa::device wrapDevice(hipDevice_t device,
                            hipCtx_t context,
                            const occa::properties &props) {

      occa::properties allProps = props;
      allProps["mode"]     = "HIP";
      allProps["device_id"] = -1;
      allProps["wrapped"]  = true;

      hip::device &dev = *(new hip::device(allProps));
      dev.dontUseRefs();

      dev.hipDevice  = device;
      dev.hipContext = context;

      dev.currentStream = dev.createStream();

      return occa::device(&dev);
    }

    occa::memory wrapMemory(occa::device device,
                            void *ptr,
                            const udim_t bytes,
                            const occa::properties &props) {

      hip::memory &mem = *(new hip::memory(props));
      mem.dontUseRefs();

      mem.modeDevice = device.getModeDevice();
      mem.ptr = (char*) ptr;
      mem.size = bytes;
      mem.mappedPtr = NULL;

      return occa::memory(&mem);
    }

    hipEvent_t& event(streamTag &tag) {
      return (hipEvent_t&) tag.modeTag;
    }

    const hipEvent_t& event(const streamTag &tag) {
      return (const hipEvent_t&) tag.modeTag;
    }

    void warn(hipError_t errorCode,
              const std::string &filename,
              const std::string &function,
              const int line,
              const std::string &message) {
      if (!errorCode) {
        return;
      }
      std::stringstream ss;
      ss << message << '\n'
         << "    Error    : HIP Error [ " << errorCode << " ]: "
         << occa::hip::getErrorMessage(errorCode);
      occa::warn(filename, function, line, ss.str());
    }

    void error(hipError_t errorCode,
               const std::string &filename,
               const std::string &function,
               const int line,
               const std::string &message) {
      if (!errorCode) {
        return;
      }
      std::stringstream ss;
      ss << message << '\n'
         << "    Error   : HIP Error [ " << errorCode << " ]: "
         << occa::hip::getErrorMessage(errorCode);
      occa::error(filename, function, line, ss.str());
    }

#define OCCA_HIP_ERROR_CASE(MACRO)              \
    case MACRO: return #MACRO

    std::string getErrorMessage(const hipError_t errorCode) {
      switch(errorCode) {
        OCCA_HIP_ERROR_CASE(hipSuccess);
        OCCA_HIP_ERROR_CASE(hipErrorInvalidValue);
        OCCA_HIP_ERROR_CASE(hipErrorMemoryAllocation);
        OCCA_HIP_ERROR_CASE(hipErrorNotInitialized);
        OCCA_HIP_ERROR_CASE(hipErrorDeinitialized);
        OCCA_HIP_ERROR_CASE(hipErrorProfilerDisabled);
        OCCA_HIP_ERROR_CASE(hipErrorProfilerNotInitialized);
        OCCA_HIP_ERROR_CASE(hipErrorProfilerAlreadyStarted);
        OCCA_HIP_ERROR_CASE(hipErrorProfilerAlreadyStopped);
        OCCA_HIP_ERROR_CASE(hipErrorNoDevice);
        OCCA_HIP_ERROR_CASE(hipErrorInvalidDevice);
        OCCA_HIP_ERROR_CASE(hipErrorInvalidImage);
        OCCA_HIP_ERROR_CASE(hipErrorInvalidContext);
        OCCA_HIP_ERROR_CASE(hipErrorContextAlreadyCurrent);
        OCCA_HIP_ERROR_CASE(hipErrorMapFailed);
        OCCA_HIP_ERROR_CASE(hipErrorUnmapFailed);
        OCCA_HIP_ERROR_CASE(hipErrorArrayIsMapped);
        OCCA_HIP_ERROR_CASE(hipErrorAlreadyMapped);
        OCCA_HIP_ERROR_CASE(hipErrorNoBinaryForGpu);
        OCCA_HIP_ERROR_CASE(hipErrorAlreadyAcquired);
        OCCA_HIP_ERROR_CASE(hipErrorNotMapped);
        OCCA_HIP_ERROR_CASE(hipErrorNotMappedAsArray);
        OCCA_HIP_ERROR_CASE(hipErrorNotMappedAsPointer);
        OCCA_HIP_ERROR_CASE(hipErrorECCNotCorrectable);
        OCCA_HIP_ERROR_CASE(hipErrorUnsupportedLimit);
        OCCA_HIP_ERROR_CASE(hipErrorContextAlreadyInUse);
        OCCA_HIP_ERROR_CASE(hipErrorPeerAccessUnsupported);
        OCCA_HIP_ERROR_CASE(hipErrorInvalidSource);
        OCCA_HIP_ERROR_CASE(hipErrorFileNotFound);
        OCCA_HIP_ERROR_CASE(hipErrorSharedObjectSymbolNotFound);
        OCCA_HIP_ERROR_CASE(hipErrorSharedObjectInitFailed);
        OCCA_HIP_ERROR_CASE(hipErrorOperatingSystem);
        OCCA_HIP_ERROR_CASE(hipErrorInvalidHandle);
        OCCA_HIP_ERROR_CASE(hipErrorNotFound);
        OCCA_HIP_ERROR_CASE(hipErrorNotReady);
        OCCA_HIP_ERROR_CASE(hipErrorLaunchOutOfResources);
        OCCA_HIP_ERROR_CASE(hipErrorLaunchTimeOut);
        OCCA_HIP_ERROR_CASE(hipErrorPeerAccessAlreadyEnabled);
        OCCA_HIP_ERROR_CASE(hipErrorPeerAccessNotEnabled);
        OCCA_HIP_ERROR_CASE(hipErrorHostMemoryAlreadyRegistered);
        OCCA_HIP_ERROR_CASE(hipErrorHostMemoryNotRegistered);

      default:
        return "UNKNOWN ERROR";
      };
    }
  }
}

#endif
