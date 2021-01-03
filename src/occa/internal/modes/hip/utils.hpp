#ifndef OCCA_INTERNAL_MODES_HIP_UTILS_HEADER
#define OCCA_INTERNAL_MODES_HIP_UTILS_HEADER

#include <occa/internal/core/device.hpp>
#include <occa/internal/modes/hip/polyfill.hpp>

namespace occa {
  namespace hip {
    typedef int advice_t;

    bool init();

    int getDeviceCount();
    hipDevice_t getDevice(const int id);
    udim_t getDeviceMemorySize(hipDevice_t device);

    std::string getVersion();

    std::string getDeviceArch(
      const int deviceId,
      const int majorVersion = -1,
      const int minorVersion = -1
    );

    void enablePeerToPeer(hipCtx_t context);
    void checkPeerToPeer(hipDevice_t destDevice,
                         hipDevice_t srcDevice);

    void peerToPeerMemcpy(hipDevice_t destDevice,
                          hipCtx_t destContext,
                          hipDeviceptr_t destMemory,
                          hipDevice_t srcDevice,
                          hipCtx_t srcContext,
                          hipDeviceptr_t srcMemory,
                          const udim_t bytes,
                          hipStream_t usingStream);


    void asyncPeerToPeerMemcpy(hipDevice_t destDevice,
                               hipCtx_t destContext,
                               hipDeviceptr_t destMemory,
                               hipDevice_t srcDevice,
                               hipCtx_t srcContext,
                               hipDeviceptr_t srcMemory,
                               const udim_t bytes,
                               hipStream_t usingStream);

    void peerToPeerMemcpy(hipDevice_t destDevice,
                          hipCtx_t destContext,
                          hipDeviceptr_t destMemory,
                          hipDevice_t srcDevice,
                          hipCtx_t srcContext,
                          hipDeviceptr_t srcMemory,
                          const udim_t bytes,
                          hipStream_t usingStream,
                          const bool isAsync);

    void advise(occa::memory mem,
                advice_t advice,
                const dim_t bytes = -1);
    void advise(occa::memory mem,
                advice_t advice,
                occa::device device);
    void advise(occa::memory mem,
                advice_t advice,
                const dim_t bytes,
                occa::device device);

    void prefetch(occa::memory mem,
                  const dim_t bytes = -1);
    void prefetch(occa::memory mem,
                  occa::device device);
    void prefetch(occa::memory mem,
                  const dim_t bytes,
                  occa::device device);

    occa::device wrapDevice(hipDevice_t device,
                            const occa::json &props = occa::json());

    void warn(hipError_t errorCode,
              const std::string &filename,
              const std::string &function,
              const int line,
              const std::string &message);

    void error(hipError_t errorCode,
               const std::string &filename,
               const std::string &function,
               const int line,
               const std::string &message);

    std::string getErrorMessage(const hipError_t errorCode);
  }
}

#endif
