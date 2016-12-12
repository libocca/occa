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

#include "occa/defines.hpp"

#if OCCA_CUDA_ENABLED

#include "occa/CUDA.hpp"

namespace occa {
  //---[ Helper Functions ]-----------
  namespace cuda {
    bool isInitialized = false;

    void init() {
      if (isInitialized) {
        return;
      }
      cuInit(0);

      isInitialized = true;
    }

    int getDeviceCount() {
      int deviceCount;
      OCCA_CUDA_CHECK("Finding Number of Devices",
                      cuDeviceGetCount(&deviceCount));
      return deviceCount;
    }

    CUdevice getDevice(const int id) {
      CUdevice device;
      OCCA_CUDA_CHECK("Getting CUdevice",
                      cuDeviceGet(&device, id));
      return device;
    }

    udim_t getDeviceMemorySize(CUdevice device) {
      size_t bytes;
      OCCA_CUDA_CHECK("Finding available memory on device",
                      cuDeviceTotalMem(&bytes, device));
      return bytes;
    }

    std::string getDeviceListInfo() {
      std::stringstream ss;

      cuda::init();
      int deviceCount = cuda::getDeviceCount();
      if (deviceCount == 0) {
        return "";
      }
      char deviceName[1024];
      OCCA_CUDA_CHECK("Getting Device Name",
                      cuDeviceGetName(deviceName, 1024, 0));

      udim_t bytes      = getDeviceMemorySize(getDevice(0));
      std::string bytesStr = stringifyBytes(bytes);

      // << "==============o=======================o==========================================\n";
      ss << "     CUDA     |  Device ID            | 0 "                                  << '\n'
         << "              |  Device Name          | " << deviceName                      << '\n'
         << "              |  Memory               | " << bytesStr                        << '\n';

      for (int i = 1; i < deviceCount; ++i) {
        bytes    = getDeviceMemorySize(getDevice(i));
        bytesStr = stringifyBytes(bytes);

        OCCA_CUDA_CHECK("Getting Device Name",
                        cuDeviceGetName(deviceName, 1024, i));

        ss << "              |-----------------------+------------------------------------------\n"
           << "              |  Device ID            | " << i                                << '\n'
           << "              |  Device Name          | " << deviceName                       << '\n'
           << "              |  Memory               | " << bytesStr                         << '\n';
      }

      return ss.str();
    }

    void enablePeerToPeer(CUcontext context) {
#if CUDA_VERSION >= 4000
      OCCA_CUDA_CHECK("Enabling Peer-to-Peer",
                      cuCtxEnablePeerAccess(context, 0) );
#else
      OCCA_CHECK(false,
                 "CUDA version ["
                 << ((int) (CUDA_VERSION / 1000))
                 << '.'
                 << ((int) ((CUDA_VERSION % 100) / 10))
                 << "] does not support Peer-to-Peer");
#endif
    }

    void checkPeerToPeer(CUdevice destDevice,
                         CUdevice srcDevice) {
#if CUDA_VERSION >= 4000
        int canAccessPeer;

        OCCA_CUDA_CHECK("Checking Peer-to-Peer Connection",
                        cuDeviceCanAccessPeer(&canAccessPeer,
                                              destDevice,
                                              srcDevice));

        OCCA_CHECK((canAccessPeer == 1),
                   "Checking Peer-to-Peer Connection");
#else
      OCCA_CHECK(false,
                 "CUDA version ["
                 << ((int) (CUDA_VERSION / 1000))
                 << '.'
                 << ((int) ((CUDA_VERSION % 100) / 10))
                 << "] does not support Peer-to-Peer");
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
        OCCA_CUDA_CHECK("Peer-to-Peer Memory Copy",
                        cuMemcpyPeer(destMemory, destContext,
                                     srcMemory , srcContext ,
                                     bytes));
      } else {
        OCCA_CUDA_CHECK("Peer-to-Peer Memory Copy",
                        cuMemcpyPeerAsync(destMemory, destContext,
                                          srcMemory , srcContext ,
                                          bytes,
                                          usingStream));
      }
#else
      OCCA_CHECK(false,
                 "CUDA version ["
                 << ((int) (CUDA_VERSION / 1000))
                 << '.'
                 << ((int) ((CUDA_VERSION % 100) / 10))
                 << "] does not support Peer-to-Peer");
#endif
    }

    occa::device wrapDevice(CUdevice device, CUcontext context) {
      occa::device dev;
      device_t<CUDA> &dHandle   = *(new device_t<CUDA>());
      CUDADeviceData_t &devData = *(new CUDADeviceData_t);

      dev.dHandle = &dHandle;

      //---[ Setup ]----------
      dHandle.data = &devData;

      devData.device     = device;
      devData.context    = context;
      devData.p2pEnabled = false;
      //======================

      dHandle.modelID_ = library::deviceModelID(dHandle.getIdentifier());
      dHandle.id_      = library::genDeviceID();

      dHandle.currentStream = dHandle.createStream();

      return dev;
    }

    CUevent& event(streamTag tag) {
      return (CUevent&) tag.handle;
    }

    std::string error(const CUresult errorCode) {
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

  const CUarray_format cudaFormats[8] = {CU_AD_FORMAT_UNSIGNED_INT8,
                                         CU_AD_FORMAT_UNSIGNED_INT16,
                                         CU_AD_FORMAT_UNSIGNED_INT32,
                                         CU_AD_FORMAT_SIGNED_INT8,
                                         CU_AD_FORMAT_SIGNED_INT16,
                                         CU_AD_FORMAT_SIGNED_INT32,
                                         CU_AD_FORMAT_HALF,
                                         CU_AD_FORMAT_FLOAT};

  template <>
  void* formatType::format<occa::CUDA>() const {
    return ((void*) &(cudaFormats[format_]));
  }

  const int CUDA_ADDRESS_NONE  = 0; // cudaBoundaryModeNone
  const int CUDA_ADDRESS_CLAMP = 1; // cudaBoundaryModeClamp
}

#endif
