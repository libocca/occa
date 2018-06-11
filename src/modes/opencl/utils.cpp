/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 David Medina and Tim Warburton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */

#include <occa/defines.hpp>

#if OCCA_OPENCL_ENABLED

#include <stdio.h>

#include <occa/modes/opencl/utils.hpp>
#include <occa/modes/opencl/device.hpp>
#include <occa/modes/opencl/memory.hpp>
#include <occa/modes/opencl/kernel.hpp>
#include <occa/io.hpp>
#include <occa/tools/sys.hpp>
#include <occa/base.hpp>

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
      cl_uint dCount;

      cl_platform_id clPID = platformID(pID);

      OCCA_OPENCL_ERROR("OpenCL: Get Device ID Count",
                        clGetDeviceIDs(clPID,
                                       deviceType(type),
                                       0, NULL, &dCount));

      return dCount;
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
                                const occa::properties &properties,
                                const io::lock_t &lock) {
      cl_int error;

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
          std::cout << "OpenCL compiling " << kernelName
                    << " from [" << sourceFile << "]";

          if (compilerFlags.size()) {
            std::cout << " with compiler flags [" << compilerFlags << "]";
          }
          std::cout << '\n';
        } else {
          std::cout << "OpenCL compiling " << kernelName << '\n';
        }
      }


      buildProgram(info,
                   kernelName,
                   compilerFlags,
                   lock);
    }

    void buildProgramFromBinary(info_t &info,
                                const std::string &source,
                                const std::string &kernelName,
                                const std::string &compilerFlags,
                                const io::lock_t &lock) {
      cl_int error, binaryError;

      const char *c_source = source.c_str();
      const size_t sourceBytes = source.size();
      info.clProgram = clCreateProgramWithBinary(info.clContext,
                                                 1, &(info.clDevice),
                                                 &sourceBytes,
                                                 (const unsigned char**) &c_source,
                                                 &binaryError, &error);

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
      cl_int error;

      error = clBuildProgram(info.clProgram,
                             1, &info.clDevice,
                             compilerFlags.c_str(),
                             NULL, NULL);

      if (error) {
        cl_int logError;
        char *log;
        size_t logSize;

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

          std::cout << "Kernel ["
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
      cl_int error;

      info.clKernel = clCreateKernel(info.clProgram,
                                     kernelName.c_str(),
                                     &error);

      if (error) {
        lock.release();
      }
      OCCA_OPENCL_ERROR("Kernel [" + kernelName + "]: Creating Kernel",
                        error);
    }

    void saveProgramBinary(cl_program &clProgram,
                           const std::string &binaryFile,
                           const io::lock_t &lock) {
      size_t binarySize;
      char *binary;

      cl_int error = clGetProgramInfo(clProgram,
                                      CL_PROGRAM_BINARY_SIZES,
                                      sizeof(size_t), &binarySize, NULL);

      if (error) {
        lock.release();
      }
      OCCA_OPENCL_ERROR("saveProgramBinary: Getting Binary Sizes",
                        error);

      binary = new char[binarySize + 1];

      error = clGetProgramInfo(clProgram,
                               CL_PROGRAM_BINARIES,
                               sizeof(char*), &binary, NULL);

      if (error) {
        lock.release();
      }
      OCCA_OPENCL_ERROR("saveProgramBinary: Getting Binary",
                        error);

      FILE *fp = fopen(binaryFile.c_str(), "wb");
      fwrite(binary, 1, binarySize, fp);
      fclose(fp);

      delete [] binary;
    }

    cl_context getCLContext(occa::device device) {
      return ((opencl::device*) device.getDHandle())->clContext;
    }

    cl_mem getCLMemory(occa::memory memory) {
      return ((opencl::memory*) memory.getMHandle())->clMem;
    }

    void* getCLMappedPtr(occa::memory memory) {
      return ((opencl::memory*) memory.getMHandle())->mappedPtr;
    }

    cl_kernel getCLKernel(occa::kernel kernel) {
      return ((opencl::kernel*) kernel.getKHandle())->clKernel;
    }

    occa::device wrapDevice(cl_device_id clDevice,
                            cl_context context,
                            const occa::properties &props) {

      occa::properties allProps = props;
      allProps["mode"]       = "OpenCL";
      allProps["platformID"] = -1;
      allProps["deviceID"]   = -1;
      allProps["wrapped"]    = true;

      opencl::device &dev = *(new opencl::device(allProps));
      dev.dontUseRefs();

      dev.platformID = -1;
      dev.deviceID   = -1;

      dev.clDevice  = clDevice;
      dev.clContext = context;

      dev.currentStream = dev.createStream();

      return occa::device(&dev);
    }

    occa::memory wrapMemory(occa::device device,
                            cl_mem clMem,
                            const udim_t bytes,
                            const occa::properties &props) {

      opencl::memory &mem = *(new opencl::memory(props));
      mem.dontUseRefs();

      mem.dHandle = device.getDHandle();
      mem.clMem   = clMem;
      mem.size    = bytes;

      return occa::memory(&mem);
    }

    cl_event& event(streamTag &tag) {
      return (cl_event&) (tag.handle);
    }

    const cl_event& event(const streamTag &tag) {
      return (const cl_event&) (tag.handle);
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
         << "    Error   : OpenCL Error [ " << clErrorCode << " ]: "
         << occa::opencl::getErrorMessage(clErrorCode);
      occa::error(filename, function, line, ss.str());
    }

    cl_int getErrorCode(cl_int errorCode) {
      errorCode = (errorCode < 0) ? errorCode : -errorCode;
      return (errorCode < 65) ? errorCode : 15;
    }

    std::string getErrorMessage(const cl_int errorCode) {
      switch(errorCode) {
      case CL_SUCCESS:                                   return "CL_SUCCESS";
      case CL_DEVICE_NOT_FOUND:                          return "CL_DEVICE_NOT_FOUND";
      case CL_DEVICE_NOT_AVAILABLE:                      return "CL_DEVICE_NOT_AVAILABLE";
      case CL_COMPILER_NOT_AVAILABLE:                    return "CL_COMPILER_NOT_AVAILABLE";
      case CL_MEM_OBJECT_ALLOCATION_FAILURE:             return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
      case CL_OUT_OF_RESOURCES:                          return "CL_OUT_OF_RESOURCES";
      case CL_OUT_OF_HOST_MEMORY:                        return "CL_OUT_OF_HOST_MEMORY";
      case CL_PROFILING_INFO_NOT_AVAILABLE:              return "CL_PROFILING_INFO_NOT_AVAILABLE";
      case CL_MEM_COPY_OVERLAP:                          return "CL_MEM_COPY_OVERLAP";
      case CL_IMAGE_FORMAT_MISMATCH:                     return "CL_IMAGE_FORMAT_MISMATCH";
      case CL_IMAGE_FORMAT_NOT_SUPPORTED:                return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
      case CL_BUILD_PROGRAM_FAILURE:                     return "CL_BUILD_PROGRAM_FAILURE";
      case CL_MAP_FAILURE:                               return "CL_MAP_FAILURE";
      case CL_MISALIGNED_SUB_BUFFER_OFFSET:              return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
      case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
      case CL_INVALID_VALUE:                             return "CL_INVALID_VALUE";
      case CL_INVALID_DEVICE_TYPE:                       return "CL_INVALID_DEVICE_TYPE";
      case CL_INVALID_PLATFORM:                          return "CL_INVALID_PLATFORM";
      case CL_INVALID_DEVICE:                            return "CL_INVALID_DEVICE";
      case CL_INVALID_CONTEXT:                           return "CL_INVALID_CONTEXT";
      case CL_INVALID_QUEUE_PROPERTIES:                  return "CL_INVALID_QUEUE_PROPERTIES";
      case CL_INVALID_COMMAND_QUEUE:                     return "CL_INVALID_COMMAND_QUEUE";
      case CL_INVALID_HOST_PTR:                          return "CL_INVALID_HOST_PTR";
      case CL_INVALID_MEM_OBJECT:                        return "CL_INVALID_MEM_OBJECT";
      case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:           return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
      case CL_INVALID_IMAGE_SIZE:                        return "CL_INVALID_IMAGE_SIZE";
      case CL_INVALID_SAMPLER:                           return "CL_INVALID_SAMPLER";
      case CL_INVALID_BINARY:                            return "CL_INVALID_BINARY";
      case CL_INVALID_BUILD_OPTIONS:                     return "CL_INVALID_BUILD_OPTIONS";
      case CL_INVALID_PROGRAM:                           return "CL_INVALID_PROGRAM";
      case CL_INVALID_PROGRAM_EXECUTABLE:                return "CL_INVALID_PROGRAM_EXECUTABLE";
      case CL_INVALID_KERNEL_NAME:                       return "CL_INVALID_KERNEL_NAME";
      case CL_INVALID_KERNEL_DEFINITION:                 return "CL_INVALID_KERNEL_DEFINITION";
      case CL_INVALID_KERNEL:                            return "CL_INVALID_KERNEL";
      case CL_INVALID_ARG_INDEX:                         return "CL_INVALID_ARG_INDEX";
      case CL_INVALID_ARG_VALUE:                         return "CL_INVALID_ARG_VALUE";
      case CL_INVALID_ARG_SIZE:                          return "CL_INVALID_ARG_SIZE";
      case CL_INVALID_KERNEL_ARGS:                       return "CL_INVALID_KERNEL_ARGS";
      case CL_INVALID_WORK_DIMENSION:                    return "CL_INVALID_WORK_DIMENSION";
      case CL_INVALID_WORK_GROUP_SIZE:                   return "CL_INVALID_WORK_GROUP_SIZE";
      case CL_INVALID_WORK_ITEM_SIZE:                    return "CL_INVALID_WORK_ITEM_SIZE";
      case CL_INVALID_GLOBAL_OFFSET:                     return "CL_INVALID_GLOBAL_OFFSET";
      case CL_INVALID_EVENT_WAIT_LIST:                   return "CL_INVALID_EVENT_WAIT_LIST";
      case CL_INVALID_EVENT:                             return "CL_INVALID_EVENT";
      case CL_INVALID_OPERATION:                         return "CL_INVALID_OPERATION";
      case CL_INVALID_GL_OBJECT:                         return "CL_INVALID_GL_OBJECT";
      case CL_INVALID_BUFFER_SIZE:                       return "CL_INVALID_BUFFER_SIZE";
      case CL_INVALID_MIP_LEVEL:                         return "CL_INVALID_MIP_LEVEL";
      case CL_INVALID_GLOBAL_WORK_SIZE:                  return "CL_INVALID_GLOBAL_WORK_SIZE";
      case CL_INVALID_PROPERTY:                          return "CL_INVALID_PROPERTY";
      default:                                           return "UNKNOWN ERROR";
      };
    }
  }
}

#endif
