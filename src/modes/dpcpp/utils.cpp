#include <stdio.h>

#include <occa/modes/dpcpp/utils.hpp>
#include <occa/modes/dpcpp/device.hpp>
#include <occa/modes/dpcpp/memory.hpp>
#include <occa/modes/dpcpp/kernel.hpp>
#include <occa/modes/dpcpp/streamTag.hpp>
#include <occa/io.hpp>
#include <occa/tools/sys.hpp>
#include <occa/core/base.hpp>

namespace occa {
  namespace dpcpp {
    info_t::info_t(){}
      dpDevice(NULL),
      dpContext(NULL),
      dpProgram(NULL),
      dpKernel(NULL) {}

    namespace info {
      std::string deviceType(int type) {
	if (type & anyType) return "ALL";
        if (type & CPU)     return "CPU";
        if (type & GPU)     return "GPU";
        if (type & FPGA)    return "FPGA";
        //if (type & XeonPhi) return "Xeon Phi";

        return "N/A";
      }

      std::string vendor(int type) {
        if (type & Intel)  return "Intel";
        if (type & AMD)    return "AMD";
        if (type & NVIDIA) return "NVIDIA";
        if (type & Altera) return "Altera";

        return "N/A";
      }
    }
/* Returns true if any DPC++ device is enabled on the machine */
    bool isEnabled() {
      auto platformlist = sycl::platform::get_platforms();
      for(auto p : platformlist)
      {
	      auto devicelist = p.get_devices(info::device_type::all);
	      if(devicelist.size() > 0)
		      return true;
      }
      return false;
    }

/* Returns the DPC++ device type*/
    sycl::info::device_type deviceType(int type) {

      if (type & info::CPU)     return sycl::info::device_type::cpu;
      if (type & info::GPU)     return sycl::info::device_type::gpu;
      if (type & info::FPGA)    return sycl::info::device_type::accelerator;

      return ret;
    }
/* Returns the number of DPC++ platforms*/
    int getPlatformCount() {
	    return sycl::platform.get_platforms().size();
    }
/* Returns the DPC++ platform of interest */
    sycl::platform* getPlatformByID(int pID) {
	    return &(sycl::platform.get_platforms()[pID]);
    }
/* Returns the number of DPC++ devices of a certain device type*/
    int getDeviceCount(int type) {
	    auto platformlist = sycl::platform.get_platforms();
	    int count = 0;
	    for(auto p : platformlist){
		    count += p.get_devices(deviceType(type)).size();
	    }
	    return count;
    }
/* Return the number of DPC++ devices under a given platform */
    int getDeviceCountInPlatform(int pID, int type) {
      return sycl::platform.get_platforms()[pID].get_devices(deviceType(type).size());
    }
/* Return the DPC++ device given the platform ID and Device ID */
    sycl::device* getDeviceByID(int pID, int dID, int type) {
	    return &(sycl::platform.get_platforms()[pID].get_devices(deviceType(type))[dID]);
    }
/* Return the DPC++ device name */
    std::string deviceName(int pID, int dID) {
	    return sycl::platform.get_platforms()[pID].get_devices(deviceType(type))[dID].get_info<info::device::name>();
    }
/* Return the DPC++ device type */
    info::device_type deviceType(int pID, int dID) {
	    return sycl::platform.get_platforms()[pID].get_devices(deviceType(type))[dID].get_info<info::device::device_type>();
    }

/* Return the DPC++ device vendor */    
    int deviceVendor(int pID, int dID) {
	    std::string devVendor = sycl::platform.get_platforms()[pID].get_devices(deviceType(type))[dID].get_info<info::device::vendors>();
	    if(devVendor.find("Intel") != std::string::npos)
		    return info::Intel;
	    else if(devVendor.find("NVIDIA") != std::string::npos)
		    return info::NVIDIA;
	    else if(devVendor.find("Altera") != std::string::npos)
		    return info::Altera;
	    else if(devVendor.find("AMD") != std::string::npos)
		   return info::AMD; 
    }
/* Returns the DPC++ Core count */
    int deviceCoreCount(int pID, int dID) {
	return sycl::platform.get_platforms()[pID].get_devices(deviceType(type))[dID].get_info<info::device::max_compute_units>();
    }
/* Returns the DPC++ global memory size given the DPC++ device */
    udim_t getDeviceMemorySize(const sycl::device &devPtr) {
	    return devPtr.get_info<info::device::global_mem_size>();
    }
/* Returns the DPC++ global memory size given the platform and device IDs */
    udim_t getDeviceMemorySize(int pID, int dID) {
	    return sycl::platform.get_platforms()[pID].get_devices(deviceType(type))[dID].get_info<info::device::global_mem_size>();
    }

    void buildProgramFromSource(info_t &info,
                                const std::string &source,
                                const std::string &kernelName,
                                const std::string &compilerFlags,
                                const std::string &sourceFile,
                                const occa::properties &properties,
                                const io::lock_t &lock) {
      cl_int error = 1;

      const bool verbose = properties.get("verbose", false);

      const char *c_source = source.c_str();
      const size_t sourceBytes = source.size();
      info.clProgram = clCreateProgramWithSource(info.clContext, 1,
                                                 &c_source,
                                                 &sourceBytes,
                                                 &error);

      if (error) {
        lock.release();
        OCCA_OPENCL_ERROR("Kernel [" + kernelName + "]: Creating Program",
                          error);
      }
      if (verbose) {
        if (lock.isInitialized()) {
          io::stdout << "OpenCL compiling " << kernelName
                     << " from [" << sourceFile << "]";

          if (compilerFlags.size()) {
            io::stdout << " with compiler flags [" << compilerFlags << "]";
          }
          io::stdout << '\n';
        } else {
          io::stdout << "OpenCL compiling " << kernelName << '\n';
        }
      }

      buildProgram(info,
                   kernelName,
                   compilerFlags,
                   lock);
    }

    void buildProgramFromBinary(info_t &info,
                                const std::string &binaryFilename,
                                const std::string &kernelName,
                                const std::string &compilerFlags,
                                const io::lock_t &lock) {
      cl_int error = 1;
      cl_int binaryError = 1;

      size_t binaryBytes;
      const char *binary = io::c_read(binaryFilename, &binaryBytes, true);
      info.clProgram = clCreateProgramWithBinary(info.clContext,
                                                 1, &(info.clDevice),
                                                 &binaryBytes,
                                                 (const unsigned char**) &binary,
                                                 &binaryError, &error);
      delete [] binary;

      if (binaryError || error) {
        lock.release();
      }
      OCCA_OPENCL_ERROR("Kernel [" + kernelName + "]: Creating Program",
                        binaryError);
      OCCA_OPENCL_ERROR("Kernel [" + kernelName + "]: Creating Program",
                        error);

      buildProgram(info,
                   kernelName,
                   compilerFlags,
                   lock);
    }

    void buildProgram(info_t &info,
                      const std::string &kernelName,
                      const std::string &compilerFlags,
                      const io::lock_t &lock) {
      cl_int error = 1;

      error = clBuildProgram(info.clProgram,
                             1, &info.clDevice,
                             compilerFlags.c_str(),
                             NULL, NULL);

      if (error) {
        cl_int logError = 1;
        char *log = NULL;
        size_t logSize = 0;

        clGetProgramBuildInfo(info.clProgram,
                              info.clDevice,
                              CL_PROGRAM_BUILD_LOG,
                              0, NULL, &logSize);

        if (logSize > 2) {
          log = new char[logSize+1];

          logError = clGetProgramBuildInfo(info.clProgram,
                                           info.clDevice,
                                           CL_PROGRAM_BUILD_LOG,
                                           logSize, log, NULL);
          OCCA_OPENCL_ERROR("Kernel [" + kernelName + "]: Building Program",
                            logError);
          log[logSize] = '\0';

          io::stderr << "Kernel ["
                     << kernelName
                     << "]: Build Log\n"
                     << log;

          delete [] log;
        }
        lock.release();
        OCCA_OPENCL_ERROR("Kernel [" + kernelName + "]: Building Program",
                          error);
      }
    }

    void buildKernelFromProgram(info_t &info,
                                const std::string &kernelName,
                                const io::lock_t &lock) {
      cl_int error = 1;

      info.clKernel = clCreateKernel(info.clProgram,
                                     kernelName.c_str(),
                                     &error);

      if (error) {
        lock.release();
      }
      OCCA_OPENCL_ERROR("Kernel [" + kernelName + "]: Creating Kernel",
                        error);
    }

    bool saveProgramBinary(info_t &info,
                           const std::string &binaryFile,
                           const io::lock_t &lock) {
      cl_int error = 1;
      cl_int binaryError = 1;

      size_t binaryBytes;
      error = clGetProgramInfo(info.clProgram,
                               CL_PROGRAM_BINARY_SIZES,
                               sizeof(size_t), &binaryBytes, NULL);
      if (error) {
        lock.release();
      }
      OCCA_OPENCL_ERROR("saveProgramBinary: Getting Binary Sizes",
                        error);

      char *binary = new char[binaryBytes + 1];
      error = clGetProgramInfo(info.clProgram,
                               CL_PROGRAM_BINARIES,
                               sizeof(char*), &binary, NULL);
      if (error) {
        lock.release();
      }
      OCCA_OPENCL_ERROR("saveProgramBinary: Getting Binary",
                        error);

      // Test to see if device supports reading from its own binary
      cl_program testProgram = clCreateProgramWithBinary(info.clContext,
                                                         1, &(info.clDevice),
                                                         &binaryBytes,
                                                         (const unsigned char**) &binary,
                                                         &binaryError, &error);

      size_t testBinaryBytes;
      error = clGetProgramInfo(testProgram,
                               CL_PROGRAM_BINARY_SIZES,
                               sizeof(size_t), &testBinaryBytes, NULL);
      if (error || !testBinaryBytes) {
        delete [] binary;
        return false;
      }

      FILE *fp = fopen(binaryFile.c_str(), "wb");
      fwrite(binary, 1, binaryBytes, fp);
      fclose(fp);

      delete [] binary;

      return true;
    }

    cl_context getCLContext(occa::device device) {
      return ((opencl::device*) device.getModeDevice())->clContext;
    }

    cl_mem getCLMemory(occa::memory memory) {
      return ((opencl::memory*) memory.getModeMemory())->clMem;
    }

    cl_kernel getCLKernel(occa::kernel kernel) {
      return ((opencl::kernel*) kernel.getModeKernel())->clKernel;
    }

    occa::device wrapDevice(cl_device_id clDevice,
                            cl_context context,
                            const occa::properties &props) {

      occa::properties allProps;
      allProps["mode"]        = "OpenCL";
      allProps["platform_id"] = -1;
      allProps["device_id"]   = -1;
      allProps["wrapped"]     = true;
      allProps += props;

      opencl::device &dev = *(new opencl::device(allProps));
      dev.dontUseRefs();

      dev.platformID = (int) allProps["platform_id"];
      dev.deviceID   = (int) allProps["device_id"];

      dev.clDevice  = clDevice;
      dev.clContext = context;

      dev.currentStream = dev.createStream(allProps["stream"]);

      return occa::device(&dev);
    }

    occa::memory wrapMemory(occa::device device,
                            void* dpcppMem,
                            const udim_t bytes,
                            const occa::properties &props) {

      dpcpp::memory &mem = *(new dpcpp::memory(device.getModeDevice(),
                                                 bytes,
                                                 props));
      mem.dontUseRefs();

      mem.dpcppMem = dpcppMem;

      return occa::memory(&mem);
    }

  }
}
