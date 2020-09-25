#ifndef OCCA_MODES_DPCPP_UTILS_HEADER
#define OCCA_MODES_DPCPP_UTILS_HEADER

#include <iostream>

#include <occa/modes/dpcpp/polyfill.hpp>
#include <occa/core/device.hpp>
#include <occa/io/lock.hpp>
#include <CL/sycl.hpp>

namespace occa {
  class streamTag;

  namespace dpcpp {

    class DPCPPFunctor{
	public:
		virtual void operator()(::sycl::nd_item<3> i){};
    };

    class info_t {
    public:
  	::sycl::device *dpcppDevice;
	::sycl::queue *dpcppQueue;
      info_t();
    };

    namespace info {
      static const int CPU     = (1 << 0);
      static const int GPU     = (1 << 1);
      static const int FPGA    = (1 << 2);
      //static const int XeonPhi = (1 << 2);
      static const int anyType = (CPU | GPU | FPGA);

      static const int Intel     = (1 << 3);
      static const int AMD       = (1 << 4);
      static const int Altera    = (1 << 5);
      static const int NVIDIA    = (1 << 6);
      static const int anyVendor = (Intel | AMD | Altera | NVIDIA);

      static const int any = (anyType | anyVendor);

      std::string deviceType(int type);
      std::string vendor(int type);
    }

    bool isEnabled();

    //cl_device_type deviceType(int type);
    ::sycl::info::device_type deviceType(int type);

    int getPlatformCount();

    ::sycl::platform getPlatformByID(int pID);


    int getDeviceCount(int type = info::anyType);
    int getDeviceCountInPlatform(int pID, int type = info::anyType);

    //cl_device_id deviceID(int pID, int dID, int type = info::any);
    ::sycl::device* deviceID(int pID, int dID, int type = info::anyType);

   /* std::string deviceStrInfo(cl_device_id clDID,
                              cl_device_info clInfo);*/

    std::string deviceName(int pID, int dID);

    int deviceType(int pID, int dID);

    int deviceVendor(int pID, int dID);

    int deviceCoreCount(int pID, int dID);

    udim_t getDeviceMemorySize(const ::sycl::device &devPtr);
    udim_t getDeviceMemorySize(int pID, int dID);

    void buildProgramFromSource(info_t &info,
                                const std::string &source,
                                const std::string &kernelName,
                                const std::string &compilerFlags = "",
                                const std::string &sourceFile = "",
                                const occa::properties &properties = occa::properties(),
                                const io::lock_t &lock = io::lock_t());

    void buildProgramFromBinary(info_t &info,
                                const std::string &binaryFilename,
                                const std::string &kernelName,
                                const std::string &compilerFlags = "",
                                const io::lock_t &lock = io::lock_t());

    void buildProgram(info_t &info,
                      const std::string &kernelName,
                      const std::string &compilerFlags,
                      const io::lock_t &lock = io::lock_t());

    void buildKernelFromProgram(info_t &info,
                                const std::string &kernelName,
                                const io::lock_t &lock = io::lock_t());

    bool saveProgramBinary(info_t &info,
                           const std::string &binaryFile,
                           const io::lock_t &lock = io::lock_t());

//    cl_context getCLContext(occa::device device);

//    cl_mem getCLMemory(occa::memory memory);

//    cl_kernel getCLKernel(occa::kernel kernel);

/*    occa::device wrapDevice(cl_device_id clDevice,
                            cl_context context,
                            const occa::properties &props = occa::properties());

    occa::memory wrapMemory(occa::device device,
                            void* dpcppMem,
                            const udim_t bytes,
                            const occa::properties &props = occa::properties());
*/
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
