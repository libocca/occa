#include <stdio.h>

#include <occa/internal/modes/opencl/utils.hpp>
#include <occa/internal/modes/opencl/device.hpp>
#include <occa/internal/modes/opencl/memory.hpp>
#include <occa/internal/modes/opencl/kernel.hpp>
#include <occa/internal/modes/opencl/streamTag.hpp>
#include <occa/internal/io.hpp>
#include <occa/internal/utils/sys.hpp>
#include <occa/core/base.hpp>

namespace occa {
  namespace opencl {
    info_t::info_t() :
      clDevice(NULL),
      clContext(NULL),
      clProgram(NULL),
      clKernel(NULL) {}

    namespace info {
      std::string deviceType(int type) {
        if (type & CPU)     return "CPU";
        if (type & GPU)     return "GPU";
        if (type & FPGA)    return "FPGA";
        if (type & XeonPhi) return "Xeon Phi";

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

    bool isEnabled() {
      cl_uint platformCount = 0;
      cl_int error = clGetPlatformIDs(0, NULL, &platformCount);
      // Only count as enabled if there is a device available
      if (!error) {
        for (cl_uint platformId = 0; platformId < platformCount; ++platformId) {
          if (getDeviceCountInPlatform(platformId)) {
            return true;
          }
        }
      }
      return false;
    }

    cl_device_type deviceType(int type) {
      cl_device_type ret = 0;

      if (type & info::CPU)     ret |= CL_DEVICE_TYPE_CPU;
      if (type & info::GPU)     ret |= CL_DEVICE_TYPE_GPU;
      if (type & info::FPGA)    ret |= CL_DEVICE_TYPE_ACCELERATOR;
      if (type & info::XeonPhi) ret |= CL_DEVICE_TYPE_ACCELERATOR;

      return ret;
    }

    int getPlatformCount() {
      cl_uint platformCount;

      OCCA_OPENCL_ERROR("OpenCL: Get Platform ID Count",
                        clGetPlatformIDs(0, NULL, &platformCount));

      return platformCount;
    }

    cl_platform_id platformID(int pID) {
      cl_platform_id *platforms = new cl_platform_id[pID + 1];

      OCCA_OPENCL_ERROR("OpenCL: Get Platform ID",
                        clGetPlatformIDs(pID + 1, platforms, NULL));

      cl_platform_id ret = platforms[pID];

      delete [] platforms;

      return ret;
    }

    int getDeviceCount(int type) {
      int pCount = opencl::getPlatformCount();
      int ret = 0;

      for (int p = 0; p < pCount; ++p)
        ret += getDeviceCountInPlatform(p, type);

      return ret;
    }

    int getDeviceCountInPlatform(int pID, int type) {
      cl_platform_id clPID = platformID(pID);
      cl_uint deviceCount = 0;
      
      clGetDeviceIDs(clPID, deviceType(type),
                     0, NULL, &deviceCount);

      return deviceCount;
    }

    cl_device_id deviceID(int pID, int dID, int type) {
      cl_device_id *devices = new cl_device_id[dID + 1];

      cl_platform_id clPID = platformID(pID);

      OCCA_OPENCL_ERROR("OpenCL: Get Device ID Count",
                        clGetDeviceIDs(clPID,
                                       deviceType(type),
                                       dID + 1, devices, NULL));

      cl_device_id ret = devices[dID];

      delete [] devices;

      return ret;
    }

    std::string deviceStrInfo(cl_device_id clDID,
                              cl_device_info clInfo) {
      size_t bytes;

      OCCA_OPENCL_ERROR("OpenCL: Getting Device String Info",
                        clGetDeviceInfo(clDID,
                                        clInfo,
                                        0, NULL, &bytes));

      char *buffer  = new char[bytes + 1];
      buffer[bytes] = '\0';

      OCCA_OPENCL_ERROR("OpenCL: Getting Device String Info",
                        clGetDeviceInfo(clDID,
                                        clInfo,
                                        bytes, buffer, NULL));

      std::string ret = buffer;

      delete [] buffer;

      size_t firstNS = ret.size();
      size_t lastNS  = ret.size();

      size_t i;

      for (i = 0; i < ret.size(); ++i) {
        if ((ret[i] != ' ') &&
            (ret[i] != '\t') &&
            (ret[i] != '\n')) {
          firstNS = i;
          break;
        }
      }

      if (i == ret.size()) {
        return "";
      }

      for (i = (ret.size() - 1); i > firstNS; --i) {
        if ((ret[i] != ' ') &&
            (ret[i] != '\t') &&
            (ret[i] != '\n')) {
          lastNS = i;
          break;
        }
      }

      if (i == firstNS) {
        return "";
      }
      return ret.substr(firstNS, (lastNS - firstNS + 1));
    }

    std::string deviceName(int pID, int dID) {
      cl_device_id clDID = deviceID(pID, dID);
      return deviceStrInfo(clDID, CL_DEVICE_NAME);
    }

    int deviceType(int pID, int dID) {
      cl_device_id clDID = deviceID(pID, dID);
      int ret = 0;

      cl_device_type clDeviceType;

      OCCA_OPENCL_ERROR("OpenCL: Get Device Type",
                        clGetDeviceInfo(clDID,
                                        CL_DEVICE_TYPE,
                                        sizeof(clDeviceType), &clDeviceType, NULL));

      if (clDeviceType & CL_DEVICE_TYPE_CPU) {
        ret |= info::CPU;
      } else if (clDeviceType & CL_DEVICE_TYPE_GPU) {
        ret |= info::GPU;
      }
      return ret;
    }

    int deviceVendor(int pID, int dID) {
      cl_device_id clDID = deviceID(pID, dID);
      int ret = 0;

      std::string vendor = deviceStrInfo(clDID, CL_DEVICE_VENDOR);

      if (vendor.find("AMD")                    != std::string::npos ||
          vendor.find("Advanced Micro Devices") != std::string::npos ||
          vendor.find("ATI")                    != std::string::npos) {

        ret |= info::AMD;
      } else if (vendor.find("Intel") != std::string::npos) {
        ret |= info::Intel;
      } else if (vendor.find("Altera") != std::string::npos) {
        ret |= info::Altera;
      } else if (vendor.find("Nvidia") != std::string::npos ||
                 vendor.find("NVIDIA") != std::string::npos) {

        ret |= info::NVIDIA;
      }

      return ret;
    }

    int deviceCoreCount(int pID, int dID) {
      cl_device_id clDID = deviceID(pID, dID);
      cl_uint ret;

      OCCA_OPENCL_ERROR("OpenCL: Get Device Core Count",
                        clGetDeviceInfo(clDID,
                                        CL_DEVICE_MAX_COMPUTE_UNITS,
                                        sizeof(ret), &ret, NULL));

      return ret;
    }

    udim_t getDeviceMemorySize(cl_device_id dID) {
      cl_ulong ret;

      OCCA_OPENCL_ERROR("OpenCL: Get Device Available Memory",
                        clGetDeviceInfo(dID,
                                        CL_DEVICE_GLOBAL_MEM_SIZE,
                                        sizeof(ret), &ret, NULL));

      return ret;
    }

    udim_t getDeviceMemorySize(int pID, int dID) {
      cl_device_id clDID = deviceID(pID, dID);

      return getDeviceMemorySize(clDID);
    }

    void buildProgramFromSource(info_t &info,
                                const std::string &source,
                                const std::string &kernelName,
                                const std::string &compilerFlags,
                                const std::string &sourceFile,
                                const occa::json &properties,
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
      const char *binary = io::c_read(binaryFilename, &binaryBytes, enums::FILE_TYPE_BINARY);
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
                            const occa::json &props) {

      occa::json allProps;
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

    void warn(cl_int errorCode,
              const std::string &filename,
              const std::string &function,
              const int line,
              const std::string &message) {
      if (!errorCode) {
        return;
      }
      const cl_int clErrorCode = getErrorCode(errorCode);
      std::stringstream ss;
      ss << message << '\n'
         << "    Error   : OpenCL Error [ " << clErrorCode << " ]: "
         << occa::opencl::getErrorMessage(clErrorCode);
      occa::warn(filename, function, line, ss.str());
    }

    void error(cl_int errorCode,
               const std::string &filename,
               const std::string &function,
               const int line,
               const std::string &message) {
      if (!errorCode) {
        return;
      }
      const cl_int clErrorCode = getErrorCode(errorCode);
      std::stringstream ss;
      ss << message << '\n'
         << "OpenCL Error [ " << clErrorCode << " ]: "
         << occa::opencl::getErrorMessage(clErrorCode);
      occa::error(filename, function, line, ss.str());
    }

    cl_int getErrorCode(cl_int errorCode) {
      errorCode = (errorCode < 0) ? errorCode : -errorCode;
      return (errorCode < 65) ? errorCode : 15;
    }

    std::string getErrorMessage(const cl_int errorCode) {
#define OCCA_OPENCL_ERROR_CASE(MACRO)           \
      case MACRO: return #MACRO

      switch(errorCode) {
        OCCA_OPENCL_ERROR_CASE(CL_SUCCESS);
        OCCA_OPENCL_ERROR_CASE(CL_DEVICE_NOT_FOUND);
        OCCA_OPENCL_ERROR_CASE(CL_DEVICE_NOT_AVAILABLE);
        OCCA_OPENCL_ERROR_CASE(CL_COMPILER_NOT_AVAILABLE);
        OCCA_OPENCL_ERROR_CASE(CL_MEM_OBJECT_ALLOCATION_FAILURE);
        OCCA_OPENCL_ERROR_CASE(CL_OUT_OF_RESOURCES);
        OCCA_OPENCL_ERROR_CASE(CL_OUT_OF_HOST_MEMORY);
        OCCA_OPENCL_ERROR_CASE(CL_PROFILING_INFO_NOT_AVAILABLE);
        OCCA_OPENCL_ERROR_CASE(CL_MEM_COPY_OVERLAP);
        OCCA_OPENCL_ERROR_CASE(CL_IMAGE_FORMAT_MISMATCH);
        OCCA_OPENCL_ERROR_CASE(CL_IMAGE_FORMAT_NOT_SUPPORTED);
        OCCA_OPENCL_ERROR_CASE(CL_BUILD_PROGRAM_FAILURE);
        OCCA_OPENCL_ERROR_CASE(CL_MAP_FAILURE);
        OCCA_OPENCL_ERROR_CASE(CL_MISALIGNED_SUB_BUFFER_OFFSET);
        OCCA_OPENCL_ERROR_CASE(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
        OCCA_OPENCL_ERROR_CASE(CL_INVALID_VALUE);
        OCCA_OPENCL_ERROR_CASE(CL_INVALID_DEVICE_TYPE);
        OCCA_OPENCL_ERROR_CASE(CL_INVALID_PLATFORM);
        OCCA_OPENCL_ERROR_CASE(CL_INVALID_DEVICE);
        OCCA_OPENCL_ERROR_CASE(CL_INVALID_CONTEXT);
        OCCA_OPENCL_ERROR_CASE(CL_INVALID_QUEUE_PROPERTIES);
        OCCA_OPENCL_ERROR_CASE(CL_INVALID_COMMAND_QUEUE);
        OCCA_OPENCL_ERROR_CASE(CL_INVALID_HOST_PTR);
        OCCA_OPENCL_ERROR_CASE(CL_INVALID_MEM_OBJECT);
        OCCA_OPENCL_ERROR_CASE(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
        OCCA_OPENCL_ERROR_CASE(CL_INVALID_IMAGE_SIZE);
        OCCA_OPENCL_ERROR_CASE(CL_INVALID_SAMPLER);
        OCCA_OPENCL_ERROR_CASE(CL_INVALID_BINARY);
        OCCA_OPENCL_ERROR_CASE(CL_INVALID_BUILD_OPTIONS);
        OCCA_OPENCL_ERROR_CASE(CL_INVALID_PROGRAM);
        OCCA_OPENCL_ERROR_CASE(CL_INVALID_PROGRAM_EXECUTABLE);
        OCCA_OPENCL_ERROR_CASE(CL_INVALID_KERNEL_NAME);
        OCCA_OPENCL_ERROR_CASE(CL_INVALID_KERNEL_DEFINITION);
        OCCA_OPENCL_ERROR_CASE(CL_INVALID_KERNEL);
        OCCA_OPENCL_ERROR_CASE(CL_INVALID_ARG_INDEX);
        OCCA_OPENCL_ERROR_CASE(CL_INVALID_ARG_VALUE);
        OCCA_OPENCL_ERROR_CASE(CL_INVALID_ARG_SIZE);
        OCCA_OPENCL_ERROR_CASE(CL_INVALID_KERNEL_ARGS);
        OCCA_OPENCL_ERROR_CASE(CL_INVALID_WORK_DIMENSION);
        OCCA_OPENCL_ERROR_CASE(CL_INVALID_WORK_GROUP_SIZE);
        OCCA_OPENCL_ERROR_CASE(CL_INVALID_WORK_ITEM_SIZE);
        OCCA_OPENCL_ERROR_CASE(CL_INVALID_GLOBAL_OFFSET);
        OCCA_OPENCL_ERROR_CASE(CL_INVALID_EVENT_WAIT_LIST);
        OCCA_OPENCL_ERROR_CASE(CL_INVALID_EVENT);
        OCCA_OPENCL_ERROR_CASE(CL_INVALID_OPERATION);
        OCCA_OPENCL_ERROR_CASE(CL_INVALID_GL_OBJECT);
        OCCA_OPENCL_ERROR_CASE(CL_INVALID_BUFFER_SIZE);
        OCCA_OPENCL_ERROR_CASE(CL_INVALID_MIP_LEVEL);
        OCCA_OPENCL_ERROR_CASE(CL_INVALID_GLOBAL_WORK_SIZE);
        OCCA_OPENCL_ERROR_CASE(CL_INVALID_PROPERTY);
        default:
          return "UNKNOWN ERROR";
      };

#undef OCCA_OPENCL_ERROR_CASE
    }
  }
}
