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

#include "occa/defines.hpp"

#if OCCA_CUDA_ENABLED

#include "occa/device.hpp"
#include "occa/tools/string.hpp"
#include "occa/tools/sys.hpp"
#include "occa/modes/cuda/device.hpp"
#include "occa/modes/cuda/utils.hpp"

namespace occa {
  //---[ Helper Functions ]-----------
  namespace cuda {
    void init() {
      static bool isInitialized = false;
      if (!isInitialized) {
        cuInit(0);
        isInitialized = true;
      }
    }

    int getDeviceCount() {
      int deviceCount;
      OCCA_CUDA_ERROR("Finding Number of Devices",
                      cuDeviceGetCount(&deviceCount));
      return deviceCount;
    }

    CUdevice getDevice(const int id) {
      CUdevice device;
      OCCA_CUDA_ERROR("Getting CUdevice",
                      cuDeviceGet(&device, id));
      return device;
    }

    udim_t getDeviceMemorySize(CUdevice device) {
      size_t bytes;
      OCCA_CUDA_ERROR("Finding available memory on device",
                      cuDeviceTotalMem(&bytes, device));
      return bytes;
    }

    void enablePeerToPeer(CUcontext context) {
#if CUDA_VERSION >= 4000
      OCCA_CUDA_ERROR("Enabling Peer-to-Peer",
                      cuCtxEnablePeerAccess(context, 0) );
#else
      OCCA_ERROR("CUDA version ["
                 << ((int) (CUDA_VERSION / 1000))
                 << '.'
                 << ((int) ((CUDA_VERSION % 100) / 10))
                 << "] does not support Peer-to-Peer",
                 false);
#endif
    }

    void checkPeerToPeer(CUdevice destDevice,
                         CUdevice srcDevice) {
#if CUDA_VERSION >= 4000
      int canAccessPeer;

      OCCA_CUDA_ERROR("Checking Peer-to-Peer Connection",
                      cuDeviceCanAccessPeer(&canAccessPeer,
                                            destDevice,
                                            srcDevice));

      OCCA_ERROR("Checking Peer-to-Peer Connection",
                 (canAccessPeer == 1));
#else
      OCCA_ERROR("CUDA version ["
                 << ((int) (CUDA_VERSION / 1000))
                 << '.'
                 << ((int) ((CUDA_VERSION % 100) / 10))
                 << "] does not support Peer-to-Peer",
                 false);
#endif
    }

    void peerToPeerMemcpy(CUdevice destDevice,
                          CUcontext destContext,
                          CUdeviceptr destMemory,
                          CUdevice srcDevice,
                          CUcontext srcContext,
                          CUdeviceptr srcMemory,
                          const udim_t bytes,
                          CUstream usingStream) {

      peerToPeerMemcpy(destDevice, destContext, destMemory,
                       srcDevice , srcContext , srcMemory ,
                       bytes,
                       usingStream, false);
    }


    void asyncPeerToPeerMemcpy(CUdevice destDevice,
                               CUcontext destContext,
                               CUdeviceptr destMemory,
                               CUdevice srcDevice,
                               CUcontext srcContext,
                               CUdeviceptr srcMemory,
                               const udim_t bytes,
                               CUstream usingStream) {

      peerToPeerMemcpy(destDevice, destContext, destMemory,
                       srcDevice , srcContext , srcMemory ,
                       bytes,
                       usingStream, true);
    }

    void peerToPeerMemcpy(CUdevice destDevice,
                          CUcontext destContext,
                          CUdeviceptr destMemory,
                          CUdevice srcDevice,
                          CUcontext srcContext,
                          CUdeviceptr srcMemory,
                          const udim_t bytes,
                          CUstream usingStream,
                          const bool isAsync) {

#if CUDA_VERSION >= 4000
      if (!isAsync) {
        OCCA_CUDA_ERROR("Peer-to-Peer Memory Copy",
                        cuMemcpyPeer(destMemory, destContext,
                                     srcMemory , srcContext ,
                                     bytes));
      } else {
        OCCA_CUDA_ERROR("Peer-to-Peer Memory Copy",
                        cuMemcpyPeerAsync(destMemory, destContext,
                                          srcMemory , srcContext ,
                                          bytes,
                                          usingStream));
      }
#else
      OCCA_ERROR("CUDA version ["
                 << ((int) (CUDA_VERSION / 1000))
                 << '.'
                 << ((int) ((CUDA_VERSION % 100) / 10))
                 << "] does not support Peer-to-Peer",
                 false);
#endif
    }

    occa::device wrapDevice(CUdevice device,
                            CUcontext context,
                            const occa::properties &props) {
      cuda::device &cdev = *(new cuda::device(props));
      cdev.handle     = device;
      cdev.context    = context;
      cdev.p2pEnabled = false;

      cdev.currentStream = cdev.createStream();

      return occa::device(&cdev);
    }

    CUevent& event(streamTag &tag) {
      return (CUevent&) tag.handle;
    }

    const CUevent& event(const streamTag &tag) {
      return (const CUevent&) tag.handle;
    }

    void warn(CUresult errorCode,
              const std::string &filename,
              const std::string &function,
              const int line,
              const std::string &message) {
      if (!errorCode) {
        return;
      }
      std::stringstream ss;
      ss << message << '\n'
         << "    Error   : CUDA Error [ " << errorCode << " ]: "
         << occa::cuda::getErrorMessage(errorCode);
      occa::warn(filename, function, line, ss.str());
    }

    void error(CUresult errorCode,
               const std::string &filename,
               const std::string &function,
               const int line,
               const std::string &message) {
      if (!errorCode) {
        return;
      }
      std::stringstream ss;
      ss << message << '\n'
         << "    Error   : CUDA Error [ " << errorCode << " ]: "
         << occa::cuda::getErrorMessage(errorCode);
      occa::error(filename, function, line, ss.str());
    }

    std::string getErrorMessage(const CUresult errorCode) {
      switch(errorCode) {
      case CUDA_SUCCESS:                              return "CUDA_SUCCESS";
      case CUDA_ERROR_INVALID_VALUE:                  return "CUDA_ERROR_INVALID_VALUE";
      case CUDA_ERROR_OUT_OF_MEMORY:                  return "CUDA_ERROR_OUT_OF_MEMORY";
      case CUDA_ERROR_NOT_INITIALIZED:                return "CUDA_ERROR_NOT_INITIALIZED";
      case CUDA_ERROR_DEINITIALIZED:                  return "CUDA_ERROR_DEINITIALIZED";
      case CUDA_ERROR_PROFILER_DISABLED:              return "CUDA_ERROR_PROFILER_DISABLED";
      case CUDA_ERROR_PROFILER_NOT_INITIALIZED:       return "CUDA_ERROR_PROFILER_NOT_INITIALIZED";
      case CUDA_ERROR_PROFILER_ALREADY_STARTED:       return "CUDA_ERROR_PROFILER_ALREADY_STARTED";
      case CUDA_ERROR_PROFILER_ALREADY_STOPPED:       return "CUDA_ERROR_PROFILER_ALREADY_STOPPED";
      case CUDA_ERROR_NO_DEVICE:                      return "CUDA_ERROR_NO_DEVICE";
      case CUDA_ERROR_INVALID_DEVICE:                 return "CUDA_ERROR_INVALID_DEVICE";
      case CUDA_ERROR_INVALID_IMAGE:                  return "CUDA_ERROR_INVALID_IMAGE";
      case CUDA_ERROR_INVALID_CONTEXT:                return "CUDA_ERROR_INVALID_CONTEXT";
      case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:        return "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";
      case CUDA_ERROR_MAP_FAILED:                     return "CUDA_ERROR_MAP_FAILED";
      case CUDA_ERROR_UNMAP_FAILED:                   return "CUDA_ERROR_UNMAP_FAILED";
      case CUDA_ERROR_ARRAY_IS_MAPPED:                return "CUDA_ERROR_ARRAY_IS_MAPPED";
      case CUDA_ERROR_ALREADY_MAPPED:                 return "CUDA_ERROR_ALREADY_MAPPED";
      case CUDA_ERROR_NO_BINARY_FOR_GPU:              return "CUDA_ERROR_NO_BINARY_FOR_GPU";
      case CUDA_ERROR_ALREADY_ACQUIRED:               return "CUDA_ERROR_ALREADY_ACQUIRED";
      case CUDA_ERROR_NOT_MAPPED:                     return "CUDA_ERROR_NOT_MAPPED";
      case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:            return "CUDA_ERROR_NOT_MAPPED_AS_ARRAY";
      case CUDA_ERROR_NOT_MAPPED_AS_POINTER:          return "CUDA_ERROR_NOT_MAPPED_AS_POINTER";
      case CUDA_ERROR_ECC_UNCORRECTABLE:              return "CUDA_ERROR_ECC_UNCORRECTABLE";
      case CUDA_ERROR_UNSUPPORTED_LIMIT:              return "CUDA_ERROR_UNSUPPORTED_LIMIT";
      case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:         return "CUDA_ERROR_CONTEXT_ALREADY_IN_USE";
      case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:        return "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED";
      case CUDA_ERROR_INVALID_SOURCE:                 return "CUDA_ERROR_INVALID_SOURCE";
      case CUDA_ERROR_FILE_NOT_FOUND:                 return "CUDA_ERROR_FILE_NOT_FOUND";
      case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND: return "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND";
      case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:      return "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED";
      case CUDA_ERROR_OPERATING_SYSTEM:               return "CUDA_ERROR_OPERATING_SYSTEM";
      case CUDA_ERROR_INVALID_HANDLE:                 return "CUDA_ERROR_INVALID_HANDLE";
      case CUDA_ERROR_NOT_FOUND:                      return "CUDA_ERROR_NOT_FOUND";
      case CUDA_ERROR_NOT_READY:                      return "CUDA_ERROR_NOT_READY";
      case CUDA_ERROR_LAUNCH_FAILED:                  return "CUDA_ERROR_LAUNCH_FAILED";
      case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:        return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";
      case CUDA_ERROR_LAUNCH_TIMEOUT:                 return "CUDA_ERROR_LAUNCH_TIMEOUT";
      case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:  return "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING";
      case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:    return "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED";
      case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:        return "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED";
      case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:         return "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE";
      case CUDA_ERROR_CONTEXT_IS_DESTROYED:           return "CUDA_ERROR_CONTEXT_IS_DESTROYED";
      case CUDA_ERROR_ASSERT:                         return "CUDA_ERROR_ASSERT";
      case CUDA_ERROR_TOO_MANY_PEERS:                 return "CUDA_ERROR_TOO_MANY_PEERS";
      case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED: return "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED";
      case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:     return "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED";
      case CUDA_ERROR_NOT_PERMITTED:                  return "CUDA_ERROR_NOT_PERMITTED";
      case CUDA_ERROR_NOT_SUPPORTED:                  return "CUDA_ERROR_NOT_SUPPORTED";
      default:                                        return "UNKNOWN ERROR";
      };
    }
  }
}

#endif
