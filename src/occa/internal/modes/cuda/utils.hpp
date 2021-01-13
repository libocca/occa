#ifndef OCCA_INTERNAL_MODES_CUDA_UTILS_HEADER
#define OCCA_INTERNAL_MODES_CUDA_UTILS_HEADER

#include <occa/internal/core/device.hpp>
#include <occa/internal/modes/cuda/polyfill.hpp>

namespace occa {
  namespace cuda {
#if CUDA_VERSION >= 8000
    typedef CUmem_advise advice_t;
#else
    typedef int advice_t;
#endif

    bool init();

    int getDeviceCount();
    CUdevice getDevice(const int id);
    udim_t getDeviceMemorySize(CUdevice device);

    std::string getVersion();

    void enablePeerToPeer(CUcontext context);
    void checkPeerToPeer(CUdevice destDevice,
                         CUdevice srcDevice);

    void peerToPeerMemcpy(CUdevice destDevice,
                          CUcontext destContext,
                          CUdeviceptr destMemory,
                          CUdevice srcDevice,
                          CUcontext srcContext,
                          CUdeviceptr srcMemory,
                          const udim_t bytes,
                          CUstream usingStream);


    void asyncPeerToPeerMemcpy(CUdevice destDevice,
                               CUcontext destContext,
                               CUdeviceptr destMemory,
                               CUdevice srcDevice,
                               CUcontext srcContext,
                               CUdeviceptr srcMemory,
                               const udim_t bytes,
                               CUstream usingStream);

    void peerToPeerMemcpy(CUdevice destDevice,
                          CUcontext destContext,
                          CUdeviceptr destMemory,
                          CUdevice srcDevice,
                          CUcontext srcContext,
                          CUdeviceptr srcMemory,
                          const udim_t bytes,
                          CUstream usingStream,
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

    CUcontext getContext(occa::device device);

    occa::device wrapDevice(CUdevice device,
                            CUcontext context,
                            const occa::json &props = occa::json());

    void warn(CUresult errorCode,
              const std::string &filename,
              const std::string &function,
              const int line,
              const std::string &message);

    void error(CUresult errorCode,
               const std::string &filename,
               const std::string &function,
               const int line,
               const std::string &message);

    void destructorError(CUresult errorCode,
                         const std::string &filename,
                         const std::string &function,
                         const int line,
                         const std::string &message);

    std::string getErrorMessage(const CUresult errorCode);
  }
}

#endif
