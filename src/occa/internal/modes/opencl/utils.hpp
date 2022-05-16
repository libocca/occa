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

    namespace info {
      enum class device_type {
        cpu, gpu, accelerator, all = cpu | gpu | accelerator
      };
    }

    bool isEnabled();

    int getPlatformCount();
    cl_platform_id platformID(int pID);

    std::string platformStrInfo(cl_platform_id clPID, cl_platform_info clInfo);
    std::string platformName(int pID);
    std::string platformVendor(int pID);
    std::string platformVersion(int pID);

    int getDeviceCount(info::device_type deviceType = info::device_type::all);
    int getDeviceCountInPlatform(int pID, info::device_type type = info::device_type::all);

    cl_device_id deviceID(int pID, int dID, info::device_type deviceType = info::device_type::all);

    std::string deviceStrInfo(cl_device_id clDID, cl_device_info clInfo);
    std::string deviceName(int pID, int dID);
    std::string deviceVendor(int pID, int dID);
    std::string deviceVersion(int pID, int dID);

    cl_device_type deviceType(info::device_type type);
    info::device_type deviceType(int pID, int dID);

    int deviceCoreCount(int pID, int dID);

    udim_t deviceGlobalMemSize(cl_device_id dID);
    udim_t deviceGlobalMemSize(int pID, int dID);

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

    cl_context getCLContext(occa::device device);

    cl_mem getCLMemory(occa::memory memory);

    cl_kernel getCLKernel(occa::kernel kernel);

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
