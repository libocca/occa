#include <occa/defines.hpp>

#include <occa/internal/core/device.hpp>
#include <occa/internal/core/memory.hpp>
#include <occa/internal/utils/string.hpp>
#include <occa/internal/utils/sys.hpp>
#include <occa/internal/modes/cuda/device.hpp>
#include <occa/internal/modes/cuda/memory.hpp>
#include <occa/internal/modes/cuda/utils.hpp>

namespace occa {
  //---[ Helper Functions ]-----------
  namespace cuda {
    bool init() {
      static bool isInitialized = false;
      if (!isInitialized) {
        isInitialized = !cuInit(0);
      }
      return isInitialized;
    }

    int getDeviceCount() {
      init();

      int deviceCount = 0;
      OCCA_CUDA_ERROR("Finding Number of Devices",
                      cuDeviceGetCount(&deviceCount));
      return deviceCount;
    }

    CUdevice getDevice(const int id) {
      init();

      CUdevice device = -1;
      OCCA_CUDA_ERROR("Getting CUdevice",
                      cuDeviceGet(&device, id));
      return device;
    }

    udim_t getDeviceMemorySize(CUdevice device) {
      size_t bytes = 0;
      OCCA_CUDA_ERROR("Finding available memory on device",
                      cuDeviceTotalMem(&bytes, device));
      return bytes;
    }

    std::string getVersion() {
      std::stringstream ss;
      ss << ((int) (CUDA_VERSION / 1000))
         << '.'
         << ((int) ((CUDA_VERSION % 100) / 10));
      return ss.str();
    }

    void enablePeerToPeer(CUcontext context) {
#if CUDA_VERSION >= 4000
      OCCA_CUDA_ERROR("Enabling Peer-to-Peer",
                      cuCtxEnablePeerAccess(context, 0) );
#else
      OCCA_FORCE_ERROR("CUDA version ["
                       << cuda::getVersion()
                       << "] does not support Peer-to-Peer");
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
                 << cuda::getVersion()
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
      OCCA_FORCE_ERROR("CUDA version ["
                       << cuda::getVersion()
                       << "] does not support Peer-to-Peer");
#endif
    }

    void advise(occa::memory mem, advice_t advice, const dim_t bytes) {
      advise(mem, advice, bytes, mem.getDevice());
    }

    void advise(occa::memory mem, advice_t advice, occa::device device) {
      advise(mem, advice, -1, device);
    }

    void advise(occa::memory mem, advice_t advice, const dim_t bytes, occa::device device) {
#if CUDA_VERSION >= 8000
      udim_t bytes_ = ((bytes == -1) ? mem.size() : bytes);
      CUdevice cuDevice = ((device.mode() == "CUDA")
                           ? ((cuda::device*) device.getModeDevice())->cuDevice
                           : CU_DEVICE_CPU);

      OCCA_CUDA_ERROR("Advising about unified memory",
                      cuMemAdvise(((cuda::memory*) mem.getModeMemory())->cuPtr,
                                  (size_t) bytes_,
                                  advice,
                                  cuDevice));
#else
      OCCA_FORCE_ERROR("CUDA version ["
                       << cuda::getVersion()
                       << "] does not support unified memory advising");
#endif
    }

    void prefetch(occa::memory mem, const dim_t bytes) {
      prefetch(mem, bytes, mem.getDevice());
    }

    void prefetch(occa::memory mem, occa::device device) {
      prefetch(mem, -1, device);
    }

    void prefetch(occa::memory mem, const dim_t bytes, occa::device device) {
      OCCA_ERROR("Memory allocated with mode [" << mem.mode() << "], not [CUDA]",
                 mem.mode() == "CUDA");

#if CUDA_VERSION >= 8000
      udim_t bytes_ = ((bytes == -1) ? mem.size() : bytes);
      CUdevice cuDevice = ((device.mode() == "CUDA")
                           ? ((cuda::device*) device.getModeDevice())->cuDevice
                           : CU_DEVICE_CPU);
      occa::stream stream = device.getStream();
      OCCA_CUDA_ERROR("Prefetching unified memory",
                      cuMemPrefetchAsync(((cuda::memory*) mem.getModeMemory())->cuPtr,
                                         (size_t) bytes_,
                                         cuDevice,
                                         *((CUstream*) stream.getModeStream())) );
#else
      OCCA_FORCE_ERROR("CUDA version ["
                       << cuda::getVersion()
                       << "] does not support unified memory prefetching");
#endif
    }

    CUcontext getContext(occa::device device) {
      return ((cuda::device*) device.getModeDevice())->cuContext;
    }

    occa::device wrapDevice(CUdevice device,
                            CUcontext context,
                            const occa::json &props) {
      occa::json allProps;
      allProps["mode"]      = "CUDA";
      allProps["device_id"] = -1;
      allProps["wrapped"]   = true;
      allProps += props;

      cuda::device &dev = *(new cuda::device(allProps));
      dev.dontUseRefs();

      dev.cuDevice  = device;
      dev.cuContext = context;

      dev.currentStream = dev.createStream(allProps["stream"]);

      return occa::device(&dev);
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
         << "    Error    : CUDA Error [ " << errorCode << " ]: "
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
         << "CUDA Error [ " << errorCode << " ]: "
         << occa::cuda::getErrorMessage(errorCode);
      occa::error(filename, function, line, ss.str());
    }

    // On destructors, ignore the case when CUDA becomes uninitialized
    void destructorError(CUresult errorCode,
                         const std::string &filename,
                         const std::string &function,
                         const int line,
                         const std::string &message) {
      if (!errorCode || errorCode == CUDA_ERROR_DEINITIALIZED) {
        return;
      }
      std::stringstream ss;
      ss << message << '\n'
         << "CUDA Error [ " << errorCode << " ]: "
         << occa::cuda::getErrorMessage(errorCode);
      occa::error(filename, function, line, ss.str());
    }

    std::string getErrorMessage(const CUresult errorCode) {
#define OCCA_CUDA_ERROR_CASE(MACRO)             \
      case MACRO: return #MACRO

      switch(errorCode) {
        OCCA_CUDA_ERROR_CASE(CUDA_SUCCESS);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_INVALID_VALUE);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_OUT_OF_MEMORY);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_NOT_INITIALIZED);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_DEINITIALIZED);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_PROFILER_DISABLED);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_PROFILER_NOT_INITIALIZED);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_PROFILER_ALREADY_STARTED);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_PROFILER_ALREADY_STOPPED);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_NO_DEVICE);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_INVALID_DEVICE);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_INVALID_IMAGE);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_INVALID_CONTEXT);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_CONTEXT_ALREADY_CURRENT);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_MAP_FAILED);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_UNMAP_FAILED);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_ARRAY_IS_MAPPED);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_ALREADY_MAPPED);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_NO_BINARY_FOR_GPU);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_ALREADY_ACQUIRED);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_NOT_MAPPED);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_NOT_MAPPED_AS_ARRAY);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_NOT_MAPPED_AS_POINTER);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_ECC_UNCORRECTABLE);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_UNSUPPORTED_LIMIT);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_CONTEXT_ALREADY_IN_USE);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_PEER_ACCESS_UNSUPPORTED);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_INVALID_SOURCE);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_FILE_NOT_FOUND);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_SHARED_OBJECT_INIT_FAILED);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_OPERATING_SYSTEM);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_INVALID_HANDLE);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_NOT_FOUND);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_NOT_READY);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_LAUNCH_FAILED);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_LAUNCH_TIMEOUT);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_PEER_ACCESS_NOT_ENABLED);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_CONTEXT_IS_DESTROYED);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_ASSERT);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_TOO_MANY_PEERS);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_NOT_PERMITTED);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_NOT_SUPPORTED);
#if CUDA_VERSION >= 7000
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_INVALID_PTX);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_ILLEGAL_ADDRESS);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_HARDWARE_STACK_ERROR);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_ILLEGAL_INSTRUCTION);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_MISALIGNED_ADDRESS);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_INVALID_ADDRESS_SPACE);
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_INVALID_PC);
#endif
#if CUDA_VERSION >= 8000
        OCCA_CUDA_ERROR_CASE(CUDA_ERROR_NVLINK_UNCORRECTABLE);
#endif
      default:
        return "UNKNOWN ERROR";
      };

#undef OCCA_CUDA_ERROR_CASE
    }
  }
}
