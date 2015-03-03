#if OCCA_OPENCL_ENABLED
#  ifndef OCCA_OPENCL_HEADER
#  define OCCA_OPENCL_HEADER

#include "occaBase.hpp"
#include "occaLibrary.hpp"

#include "occaKernelDefines.hpp"

#include "occaDefines.hpp"

#if   (OCCA_OS & LINUX_OS)
#  include <CL/cl.h>
#  include <CL/cl_gl.h>
#elif (OCCA_OS & OSX_OS)
#  include <OpenCL/OpenCl.h>
#else
#  include "CL/opencl.h"
#endif

namespace occa {
  //---[ Data Structs ]---------------
  struct OpenCLKernelData_t {
    int platform, device;

    cl_platform_id platformID;
    cl_device_id   deviceID;
    cl_context     context;
    cl_program     program;
    cl_kernel      kernel;
  };

  struct OpenCLDeviceData_t {
    int platform, device;

    cl_platform_id platformID;
    cl_device_id   deviceID;
    cl_context     context;
  };
  //==================================


  //---[ Helper Functions ]-----------
  namespace cl {
    cl_device_type deviceType(int type);

    int platformCount();

    cl_platform_id platformID(int pID);

    int deviceCount(int type = any);
    int deviceCountInPlatform(int pID, int type = any);

    cl_device_id deviceID(int pID, int dID, int type = any);

    std::string deviceStrInfo(cl_device_id clDID,
                              cl_device_info clInfo);

    std::string deviceName(int pID, int dID);

    int deviceType(int pID, int dID);

    int deviceVendor(int pID, int dID);

    int deviceCoreCount(int pID, int dID);

    std::string getDeviceListInfo();

    void buildKernelFromSource(OpenCLKernelData_t &data_,
                                const char *content,
                                const size_t contentBytes,
                                const std::string &functionName,
                                const std::string &flags = "",
                                const std::string &cachedBinary = "",
                                const std::string &iCachedBinary = "");

    void buildKernelFromBinary(OpenCLKernelData_t &data_,
                               const unsigned char *content,
                               const size_t contentBytes,
                               const std::string &functionName,
                               const std::string &flags = "");

    void saveProgramBinary(OpenCLKernelData_t &data_,
                           const std::string &cachedBinary);

    bool imageFormatIsSupported(cl_image_format &f,
                                cl_image_format *fs,
                                const int formatCount);

    void printImageFormat(cl_image_format &imageFormat);
  };

  extern const cl_channel_type clFormats[8];

  template <>
  void* formatType::format<occa::OpenCL>() const;
  //==================================


  //---[ Kernel ]---------------------
  template <>
  kernel_t<OpenCL>::kernel_t();

  template <>
  kernel_t<OpenCL>::kernel_t(const kernel_t &k);

  template <>
  kernel_t<OpenCL>& kernel_t<OpenCL>::operator = (const kernel_t<OpenCL> &k);

  template <>
  kernel_t<OpenCL>::kernel_t(const kernel_t<OpenCL> &k);

  template <>
  std::string kernel_t<OpenCL>::getCachedBinaryName(const std::string &filename,
                                                    kernelInfo &info_);

  template <>
  kernel_t<OpenCL>* kernel_t<OpenCL>::buildFromSource(const std::string &filename,
                                                      const std::string &functionName_,
                                                      const kernelInfo &info_);

  template <>
  kernel_t<OpenCL>* kernel_t<OpenCL>::buildFromBinary(const std::string &filename,
                                                      const std::string &functionName_);

  template <>
  kernel_t<OpenCL>* kernel_t<OpenCL>::loadFromLibrary(const char *cache,
                                                      const std::string &functionName_);

  template <>
  int kernel_t<OpenCL>::preferredDimSize();

  template <>
  double kernel_t<OpenCL>::timeTaken();

  template <>
  double kernel_t<OpenCL>::timeTakenBetween(void *start, void *end);

  template <>
  void kernel_t<OpenCL>::free();
  //==================================


  //---[ Memory ]---------------------
  template <>
  memory_t<OpenCL>::memory_t();

  template <>
  memory_t<OpenCL>::memory_t(const memory_t &m);

  template <>
  memory_t<OpenCL>& memory_t<OpenCL>::operator = (const memory_t &m);

  template <>
  void* memory_t<OpenCL>::getMemoryHandle();

  template <>
  void* memory_t<OpenCL>::getTextureHandle();

  template <>
  void memory_t<OpenCL>::copyFrom(const void *src,
                                  const uintptr_t bytes,
                                  const uintptr_t offset);

  template <>
  void memory_t<OpenCL>::copyFrom(const memory_v *src,
                                  const uintptr_t bytes,
                                  const uintptr_t destOffset,
                                  const uintptr_t srcOffset);

  template <>
  void memory_t<OpenCL>::copyTo(void *dest,
                                const uintptr_t bytes,
                                const uintptr_t offset);

  template <>
  void memory_t<OpenCL>::copyTo(memory_v *dest,
                                const uintptr_t bytes,
                                const uintptr_t destOffset,
                                const uintptr_t srcOffset);

  template <>
  void memory_t<OpenCL>::asyncCopyFrom(const void *src,
                                       const uintptr_t bytes,
                                       const uintptr_t destOffset);

  template <>
  void memory_t<OpenCL>::asyncCopyFrom(const memory_v *src,
                                       const uintptr_t bytes,
                                       const uintptr_t srcOffset,
                                       const uintptr_t offset);

  template <>
  void memory_t<OpenCL>::asyncCopyTo(void *dest,
                                     const uintptr_t bytes,
                                     const uintptr_t offset);

  template <>
  void memory_t<OpenCL>::asyncCopyTo(memory_v *dest,
                                     const uintptr_t bytes,
                                     const uintptr_t destOffset,
                                     const uintptr_t srcOffset);

  template <>
  void memory_t<OpenCL>::mappedFree();

  template <>
  void memory_t<OpenCL>::free();
  //==================================


  //---[ Device ]---------------------
  template <>
  device_t<OpenCL>::device_t();

  template <>
  device_t<OpenCL>::device_t(const device_t<OpenCL> &k);

  template <>
  device_t<OpenCL>& device_t<OpenCL>::operator = (const device_t<OpenCL> &k);

  template <>
  void device_t<OpenCL>::setup(argInfoMap &aim);

  template <>
  void device_t<OpenCL>::addOccaHeadersToInfo(kernelInfo &info_);

  template <>
  std::string device_t<OpenCL>::getInfoSalt(const kernelInfo &info_);

  template <>
  deviceIdentifier device_t<OpenCL>::getIdentifier() const;

  template <>
  void device_t<OpenCL>::getEnvironmentVariables();

  template <>
  void device_t<OpenCL>::appendAvailableDevices(std::vector<device> &dList);

  template <>
  void device_t<OpenCL>::setCompiler(const std::string &compiler_);

  template <>
  void device_t<OpenCL>::setCompilerEnvScript(const std::string &compilerEnvScript_);

  template <>
  void device_t<OpenCL>::setCompilerFlags(const std::string &compilerFlags_);

  template <>
  void device_t<OpenCL>::flush();

  template <>
  void device_t<OpenCL>::finish();

  template <>
  void device_t<OpenCL>::waitFor(streamTag tag);

  template <>
  stream device_t<OpenCL>::createStream();

  template <>
  void device_t<OpenCL>::freeStream(stream s);

  template <>
  stream device_t<OpenCL>::wrapStream(void *handle_);

  template <>
  streamTag device_t<OpenCL>::tagStream();

  template <>
  double device_t<OpenCL>::timeBetween(const streamTag &startTag, const streamTag &endTag);

  template <>
  kernel_v* device_t<OpenCL>::buildKernelFromSource(const std::string &filename,
                                                    const std::string &functionName_,
                                                    const kernelInfo &info_);

  template <>
  kernel_v* device_t<OpenCL>::buildKernelFromBinary(const std::string &filename,
                                                    const std::string &functionName_);

  template <>
  void device_t<OpenCL>::cacheKernelInLibrary(const std::string &filename,
                                              const std::string &functionName_,
                                              const kernelInfo &info_);

  template <>
  kernel_v* device_t<OpenCL>::loadKernelFromLibrary(const char *cache,
                                                    const std::string &functionName_);

  template <>
  memory_v* device_t<OpenCL>::wrapMemory(void *handle_,
                                         const uintptr_t bytes);

  template <>
  memory_v* device_t<OpenCL>::wrapTexture(void *handle_,
                                          const int dim, const occa::dim &dims,
                                          occa::formatType type, const int permissions);

  template <>
  memory_v* device_t<OpenCL>::malloc(const uintptr_t bytes,
                                     void *src);

  template <>
  memory_v* device_t<OpenCL>::textureAlloc(const int dim, const occa::dim &dims,
                                           void *src,
                                           occa::formatType type, const int permissions);

  template <>
  memory_v* device_t<OpenCL>::mappedAlloc(const uintptr_t bytes,
                                          void *src);

  template <>
  void device_t<OpenCL>::free();

  template <>
  int device_t<OpenCL>::simdWidth();
  //==================================

#include "operators/occaOpenCLKernelOperators.hpp"

  //---[ Error Handling ]-------------
  std::string openclError(int e);
  //==================================
};

#  endif
#endif
