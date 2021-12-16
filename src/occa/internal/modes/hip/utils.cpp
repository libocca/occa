#include <occa/internal/core/device.hpp>
#include <occa/internal/core/memory.hpp>
#include <occa/internal/utils/string.hpp>
#include <occa/internal/utils/sys.hpp>
#include <occa/internal/modes/hip/device.hpp>
#include <occa/internal/modes/hip/memory.hpp>
#include <occa/internal/modes/hip/utils.hpp>

namespace occa {
  //---[ Helper Functions ]-----------
  namespace hip {
    bool init() {
      static bool isInitialized = false;
      if (!isInitialized) {
        isInitialized = !hipInit(0);
      }
      return isInitialized;
    }

    int getDeviceCount() {
      int deviceCount = 0;
      static_cast<void>(hipGetDeviceCount(&deviceCount));
      return deviceCount;
    }

    hipDevice_t getDevice(const int id) {
      hipDevice_t device = 0;
      OCCA_HIP_ERROR("Getting hipDevice_t",
                     hipDeviceGet(&device, id));
      return device;
    }

    udim_t getDeviceMemorySize(hipDevice_t device) {
      size_t bytes = 0;
      OCCA_HIP_ERROR("Finding available memory on device",
                     hipDeviceTotalMem(&bytes, device));
      return bytes;
    }

    std::string getVersion() {
      std::stringstream ss;
      int driverVersion = 0;
      OCCA_HIP_ERROR("Finding HIP driver version",
                     hipDriverGetVersion(&driverVersion));
      ss << driverVersion;

      return ss.str();
    }

    std::string getDeviceArch(
      const int deviceId,
      const int majorVersion,
      const int minorVersion
    ) {
      hipDeviceProp_t hipProps;

      OCCA_HIP_ERROR("Getting HIP device properties",
                     hipGetDeviceProperties(&hipProps, deviceId));

      if (hipProps.gcnArch) { // AMD or NVIDIA
#if HIP_VERSION >= 306
        return hipProps.gcnArchName;
#else
        return "gfx" + toString(hipProps.gcnArch);
#endif
      }

      std::string sm = "sm_";
      sm += toString(
        majorVersion >= 0
        ? majorVersion
        : hipProps.major
      );
      sm += toString(
        minorVersion >= 0
        ? minorVersion
        : hipProps.minor
      );
      return sm;
    }

    void enablePeerToPeer(hipCtx_t context) {

      // OCCA_HIP_ERROR("Enabling Peer-to-Peer",
      //                hipCtxEnablePeerAccess(context, 0) );
    }

    void checkPeerToPeer(hipDevice_t destDevice,
                         hipDevice_t srcDevice) {
      int canAccessPeer = 0;

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

    occa::device wrapDevice(hipDevice_t device,
                            const occa::json &props) {

      occa::json allProps;
      allProps["mode"]     = "HIP";
      allProps["device_id"] = -1;
      allProps["wrapped"]  = true;
      allProps += props;

      hip::device &dev = *(new hip::device(allProps));
      dev.dontUseRefs();

      dev.hipDevice  = device;

      dev.currentStream = dev.createStream(allProps["stream"]);

      return occa::device(&dev);
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
         << "HIP Error [ " << errorCode << " ]: "
         << occa::hip::getErrorMessage(errorCode);
      occa::error(filename, function, line, ss.str());
    }

    std::string getErrorMessage(const hipError_t errorCode) {
#define OCCA_HIP_ERROR_CASE(MACRO)              \
      case MACRO: return #MACRO

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
#undef OCCA_HIP_ERROR_CASE
    }
  }
}
