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

#include "occa/modes/serial/device.hpp"
#include "occa/modes/serial/kernel.hpp"
#include "occa/modes/serial/memory.hpp"
#include "occa/tools/env.hpp"
#include "occa/tools/io.hpp"
#include "occa/tools/sys.hpp"
#include "occa/base.hpp"

namespace occa {
  namespace serial {
    device::device(const occa::properties &properties_) :
      occa::device_v(properties_) {

      int vendor;
      std::string compiler, compilerFlags, compilerEnvScript;

      if (properties.get<std::string>("compiler").size()) {
        compiler = properties["compiler"].getString();
      } else if (env::var("OCCA_CXX").size()) {
        compiler = env::var("OCCA_CXX");
      } else if (env::var("CXX").size()) {
        compiler = env::var("CXX");
      } else {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
        compiler = "g++";
#else
        compiler = "cl.exe";
#endif
      }

      vendor = sys::compilerVendor(compiler);

      if (properties.get<std::string>("compilerFlags").size()) {
        compilerFlags = properties["compilerFlags"].getString();
      } else if (env::var("OCCA_CXXFLAGS").size()) {
        compilerFlags = env::var("OCCA_CXXFLAGS");
      } else if (env::var("CXXFLAGS").size()) {
        compilerFlags = env::var("CXXFLAGS");
      } else {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
        compilerFlags = "-g";
#else
#  if OCCA_DEBUG_ENABLED
        compilerFlags = " /Od";
#  else
        compilerFlags = " /O2";
#  endif
#endif
      }

      if (properties.get<std::string>("compilerEnvScript").size()) {
        compilerEnvScript = properties["compilerEnvScript"].getString();
      } else {
#if (OCCA_OS == OCCA_WINDOWS_OS)
        std::string byteness;

        if (sizeof(void*) == 4) {
          byteness = "x86 ";
        } else if (sizeof(void*) == 8) {
          byteness = "amd64";
        } else {
          OCCA_ERROR("sizeof(void*) is not equal to 4 or 8",
                     false);
        }
#  if   (OCCA_VS_VERSION == 1800)
        // MSVC++ 12.0 - Visual Studio 2013
        char *visualStudioTools = getenv("VS120COMNTOOLS");
#  elif (OCCA_VS_VERSION == 1700)
        // MSVC++ 11.0 - Visual Studio 2012
        char *visualStudioTools = getenv("VS110COMNTOOLS");
#  else (OCCA_VS_VERSION < 1700)
        // MSVC++ 10.0 - Visual Studio 2010
        char *visualStudioTools = getenv("VS100COMNTOOLS");
#  endif

        if (visualStudioTools != NULL) {
          compilerEnvScript = "\"" + std::string(visualStudioTools) + "..\\..\\VC\\vcvarsall.bat\" " + byteness;
        } else {
          std::cout << "WARNING: Visual Studio environment variable not found -> compiler environment (vcvarsall.bat) maybe not correctly setup." << std::endl;
        }
#endif
      }

      properties["vendor"] = vendor;
      sys::addSharedBinaryFlagsTo(vendor, compilerFlags);

      properties["compiler"]          = compiler;
      properties["compilerFlags"]     = compilerFlags;
      properties["compilerEnvScript"] = compilerEnvScript;
    }

    device::~device() {}

    void* device::getHandle(const occa::properties &props) {
      return NULL;
    }

    void device::finish() {}

    bool device::hasSeparateMemorySpace() {
      return false;
    }

    hash_t device::hash() {
      if (!hash_.initialized) {
        hash_ = occa::hash(properties);
      }
      return hash_;
    }

    void device::waitFor(streamTag tag) {}

    stream_t device::createStream() {
      return NULL;
    }

    void device::freeStream(stream_t s) {}

    stream_t device::wrapStream(void *handle_) {
      return NULL;
    }

    streamTag device::tagStream() {
      streamTag ret;
      ret.tagTime = sys::currentTime();
      return ret;
    }

    double device::timeBetween(const streamTag &startTag, const streamTag &endTag) {
      return (endTag.tagTime - startTag.tagTime);
    }

    kernel_v* device::buildKernel(const std::string &filename,
                                  const std::string &functionName,
                                  const occa::properties &props) {
      kernel *k = new kernel();
      k->dHandle = this;
      k->build(filename, functionName, props);
      return k;
    }

    kernel_v* device::buildKernelFromBinary(const std::string &filename,
                                            const std::string &functionName,
                                            const occa::properties &props) {
      kernel *k = new kernel();
      k->dHandle = this;
      k->buildFromBinary(filename, functionName, props);
      return k;
    }

    memory_v* device::malloc(const udim_t bytes,
                             void *src,
                             const occa::properties &props) {
      memory *mem = new memory();

      mem->dHandle = this;
      mem->size    = bytes;
      mem->handle  = sys::malloc(bytes);

      if (src != NULL) {
        ::memcpy(mem->handle, src, bytes);
      }

      return mem;
    }

    memory_v* device::wrapMemory(void *handle_,
                                 const udim_t bytes,
                                 const occa::properties &props) {
      memory *mem = new memory();

      mem->dHandle = this;
      mem->size    = bytes;
      mem->handle  = handle_;

      return mem;
    }

    udim_t device::memorySize() {
      return sys::installedRAM();
    }

    void device::free() {}
  }
}
