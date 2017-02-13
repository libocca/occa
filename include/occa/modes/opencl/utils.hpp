/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2017 David Medina and Tim Warburton
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

#include "occa/defines.hpp"

#if OCCA_OPENCL_ENABLED
#  ifndef OCCA_OPENCL_UTILS_HEADER
#  define OCCA_OPENCL_UTILS_HEADER

#include <iostream>

#include "occa/modes/opencl/headers.hpp"
#include "occa/device.hpp"

namespace occa {
  class streamTag;

  namespace opencl {
    struct info_t {
      cl_device_id clDeviceID;
      cl_context   clContext;
      cl_program   clProgram;
      cl_kernel    clKernel;
    };

    namespace info {
      static const int CPU     = (1 << 0);
      static const int GPU     = (1 << 1);
      static const int FPGA    = (1 << 3);
      static const int XeonPhi = (1 << 2);
      static const int anyType = (CPU | GPU | FPGA | XeonPhi);

      static const int Intel     = (1 << 4);
      static const int AMD       = (1 << 5);
      static const int Altera    = (1 << 6);
      static const int NVIDIA    = (1 << 7);
      static const int anyVendor = (Intel | AMD | Altera | NVIDIA);

      static const int any = (anyType | anyVendor);

      std::string deviceType(int type);
      std::string vendor(int type);
    }

    cl_device_type deviceType(int type);

    int getPlatformCount();

    cl_platform_id platformID(int pID);

    int getDeviceCount(int type = info::any);
    int getDeviceCountInPlatform(int pID, int type = info::any);

    cl_device_id deviceID(int pID, int dID, int type = info::any);

    std::string deviceStrInfo(cl_device_id clDID,
                              cl_device_info clInfo);

    std::string deviceName(int pID, int dID);

    int deviceType(int pID, int dID);

    int deviceVendor(int pID, int dID);

    int deviceCoreCount(int pID, int dID);

    udim_t getDeviceMemorySize(cl_device_id dID);
    udim_t getDeviceMemorySize(int pID, int dID);

    void buildKernel(info_t &info_,
                     const char *content,
                     const size_t contentBytes,
                     const std::string &functionName,
                     const std::string &flags = "",
                     hash_t hash = hash_t(),
                     const std::string &sourceFile = "");

    void buildKernelFromBinary(info_t &info_,
                               const unsigned char *content,
                               const size_t contentBytes,
                               const std::string &functionName,
                               const std::string &flags = "");

    void saveProgramBinary(info_t &info_,
                           const std::string &binaryFile,
                           const hash_t &hash,
                           const std::string &hashTag);

    occa::device wrapDevice(cl_platform_id platformID,
                            cl_device_id deviceID,
                            cl_context context);

    cl_event& event(streamTag &tag);
    const cl_event& event(const streamTag &tag);

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

#  endif
#endif
