#ifndef OCCA_INTERNAL_MODES_OPENCL_UTILS_HEADER
#define OCCA_INTERNAL_MODES_OPENCL_UTILS_HEADER

#include <iostream>

#include <occa/internal/modes/opencl/polyfill.hpp>
#include <occa/internal/core/device.hpp>

namespace occa {
  class streamTag;

  namespace opencl {
    class info_t {
    public:
      cl_device_id clDevice;
      cl_context clContext;
      cl_program clProgram;
      cl_kernel clKernel;

      info_t();
    };

    constexpr cl_device_type default_device_type = (CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU);

    bool isEnabled();

    std::vector<cl_platform_id> getPlatforms(cl_device_type device_type = default_device_type);
    cl_platform_id getPlatformFromDevice(cl_device_id device_id);

    std::string platformStrInfo(cl_platform_id clPID, cl_platform_info clInfo);
    
    std::string platformName(cl_platform_id platform_id);
    std::string platformVendor(cl_platform_id platform_id);
    std::string platformVersion(cl_platform_id platform_id);

    int getDeviceCount(cl_device_type device_type = default_device_type);
    int getDeviceCountInPlatform(cl_platform_id, cl_device_type device_type = default_device_type);

    std::vector<cl_device_id> getDevicesInPlatform(cl_platform_id platform_id, cl_device_type device_type = default_device_type);

    std::string deviceStrInfo(cl_device_id clDID, cl_device_info clInfo);
    std::string deviceName(cl_device_id device_id);
    std::string deviceVendor(cl_device_id device_id);
    std::string deviceVersion(cl_device_id device_id);

    cl_device_type deviceType(cl_device_id device_type);

    int deviceCoreCount(cl_device_id device_id);

    udim_t deviceGlobalMemSize(cl_device_id dID);

    cl_context createContextFromDevice(cl_device_id device_id);

    void buildProgramFromSource(info_t &info,
                                const std::string &source,
                                const std::string &kernelName,
                                const std::string &compilerFlags = "",
                                const std::string &sourceFile = "",
                                const occa::json &properties = occa::json());

    void buildProgramFromBinary(info_t &info,
                                const std::string &binaryFilename,
                                const std::string &kernelName,
                                const std::string &compilerFlags = "");

    void buildProgram(info_t &info,
                      const std::string &kernelName,
                      const std::string &compilerFlags);

    void buildKernelFromProgram(info_t &info,
                                const std::string &kernelName);

    bool saveProgramBinary(info_t &info,
                           const std::string &binaryFile);

    occa::device wrapDevice(cl_device_id clDevice,
                            cl_context context,
                            const occa::json &props = occa::json());

    occa::memory wrapMemory(occa::device device,
                            cl_mem clMem,
                            const udim_t bytes,
                            const occa::json &props = occa::json());

    void warn(cl_int errorCode,
              const std::string &filename,
              const std::string &function,
              const int line,
              const std::string &message);

    void error(cl_int errorCode,
               const std::string &filename,
               const std::string &function,
               const int line,
               const std::string &message);

    int getErrorCode(int errorCode);
    std::string getErrorMessage(const int errorCode);
  }
}

#endif
