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

#include <occa/base.hpp>
#include <occa/tools/env.hpp>
#include <occa/io.hpp>
#include <occa/tools/sys.hpp>
#include <occa/modes/serial/device.hpp>
#include <occa/modes/serial/kernel.hpp>
#include <occa/modes/serial/memory.hpp>
#include <occa/lang/modes/serial.hpp>

namespace occa {
  namespace serial {
    device::device(const occa::properties &properties_) :
      occa::device_v(properties_) {

      int vendor;
      std::string compiler, compilerFlags, compilerEnvScript;

      if (properties.get<std::string>("kernel/compiler").size()) {
        compiler = (std::string) properties["kernel/compiler"];
      } else if (env::var("OCCA_CXX").size()) {
        compiler = env::var("OCCA_CXX");
      } else if (env::var("CXX").size()) {
        compiler = env::var("CXX");
      } else {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
        compiler = "g++";
#else
        compiler = "cl.exe";
#endif
      }

      vendor = sys::compilerVendor(compiler);

      if (properties.get<std::string>("kernel/compilerFlags").size()) {
        compilerFlags = (std::string) properties["kernel/compilerFlags"];
      } else if (env::var("OCCA_CXXFLAGS").size()) {
        compilerFlags = env::var("OCCA_CXXFLAGS");
      } else if (env::var("CXXFLAGS").size()) {
        compilerFlags = env::var("CXXFLAGS");
      } else {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
        compilerFlags = "-g";
#else
        compilerFlags = " /Od";
#endif
      }

      if (properties.get<std::string>("kernel/compilerEnvScript").size()) {
        compilerEnvScript = (std::string) properties["kernel/compilerEnvScript"];
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

      properties["kernel/vendor"] = vendor;
      sys::addSharedBinaryFlagsTo(vendor, compilerFlags);

      properties["kernel/compiler"]          = compiler;
      properties["kernel/compilerFlags"]     = compilerFlags;
      properties["kernel/compilerEnvScript"] = compilerEnvScript;
    }

    device::~device() {}

    void device::free() {}

    void device::finish() const {}

    bool device::hasSeparateMemorySpace() const {
      return false;
    }

    hash_t device::hash() const {
      if (!hash_.initialized) {
        hash_ = occa::hash("host");
      }
      return hash_;
    }

    //---[ Stream ]---------------------
    stream_t device::createStream() const {
      return NULL;
    }

    void device::freeStream(stream_t s) const {}

    streamTag device::tagStream() const {
      streamTag ret;
      ret.tagTime = sys::currentTime();
      return ret;
    }

    void device::waitFor(streamTag tag) const {}

    double device::timeBetween(const streamTag &startTag,
                               const streamTag &endTag) const {
      return (endTag.tagTime - startTag.tagTime);
    }

    stream_t device::wrapStream(void *handle_,
                                const occa::properties &props) const {
      return NULL;
    }
    //==================================

    //---[ Kernel ]---------------------
    bool device::parseFile(const std::string &filename,
                           const std::string &outputFile,
                           const occa::properties &kernelProps) {
      lang::okl::serialParser parser(kernelProps);
      parser.parseFile(filename);

      // Verify if parsing succeeded
      if (!parser.succeeded()) {
        if (!kernelProps.get("silent", false)) {
          OCCA_FORCE_ERROR("Unable to transform OKL kernel");
        }
        return false;
      }

      if (!sys::fileExists(outputFile)) {
        hash_t hash = occa::hash(outputFile);
        io::lock_t lock(hash, "serial-parser");
        if (lock.isMine()) {
          parser.writeToFile(outputFile);
        }
      }

      return true;
    }

    kernel_v* device::buildKernel(const std::string &filename,
                                  const std::string &kernelName,
                                  const hash_t kernelHash,
                                  const occa::properties &kernelProps) {
      const std::string hashDir = io::hashDir(filename, kernelHash);
       std::string binaryFilename = hashDir + kc::binaryFile;
      bool foundBinary = true;

      // This is a launcher kernel
      // TODO: Clean this up
      if (startsWith(filename, hashDir)) {
        binaryFilename = io::dirname(filename) + kc::hostBinaryFile;
      }

      io::lock_t lock(kernelHash, "serial-kernel");
      if (lock.isMine()) {
        if (sys::fileExists(binaryFilename)) {
          lock.release();
        } else {
          foundBinary = false;
        }
      }

      const bool verbose = kernelProps.get("verbose", false);
      if (foundBinary) {
        if (verbose) {
          std::cout << "Loading cached ["
                    << kernelName
                    << "] from ["
                    << io::shortname(filename)
                    << "] in [" << io::shortname(binaryFilename) << "]\n";
        }
        return buildKernelFromBinary(binaryFilename,
                                     kernelName,
                                     kernelProps);
      }

      // Cache raw origin
      std::string sourceFilename = (
        io::cacheFile(filename,
                      kc::rawSourceFile,
                      kernelHash,
                      assembleHeader(kernelProps))
      );

      if (kernelProps.get("okl", true)) {
        const std::string outputFile = hashDir + kc::sourceFile;
        bool valid = parseFile(sourceFilename,
                               outputFile,
                               kernelProps);
        if (!valid) {
          return NULL;
        }
        sourceFilename = outputFile;
      }

      std::stringstream command;
      std::string compilerEnvScript = kernelProps["compilerEnvScript"];
      if (compilerEnvScript.size()) {
        command << compilerEnvScript << " && ";
      }

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
      command << (std::string) kernelProps["compiler"]
              << ' '    << (std::string) kernelProps["compilerFlags"]
              << ' '    << sourceFilename
              << " -o " << binaryFilename
              << " -I"  << env::OCCA_DIR << "include"
              << " -L"  << env::OCCA_DIR << "lib -locca"
              << std::endl;
#else
      command << kernelProps["compiler"]
              << " /D MC_CL_EXE"
              << " /D OCCA_OS=OCCA_WINDOWS_OS"
              << " /EHsc"
              << " /wd4244 /wd4800 /wd4804 /wd4018"
              << ' '       << kernelProps["compilerFlags"]
              << " /I"     << env::OCCA_DIR << "include"
              << ' '       << sourceFilename
              << " /link " << env::OCCA_DIR << "lib/libocca.lib",
              << " /OUT:"  << binaryFilename
              << std::endl;
#endif

      const std::string &sCommand = command.str();

      if (verbose) {
        std::cout << "Compiling [" << kernelName << "]\n" << sCommand << "\n";
      }

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
      const int compileError = system(sCommand.c_str());
#else
      const int compileError = system(("\"" +  sCommand + "\"").c_str());
#endif

      lock.release();
      if (compileError) {
        OCCA_FORCE_ERROR("Error compiling [" << kernelName << "],"
                         " Command: [" << sCommand << ']');
      }

      kernel_v *k = buildKernelFromBinary(binaryFilename,
                                          kernelName,
                                          kernelProps);
      if (k) {
        k->sourceFilename = sourceFilename;
      }
      return k;
    }

    kernel_v* device::buildKernelFromBinary(const std::string &filename,
                                            const std::string &kernelName,
                                            const occa::properties &kernelProps) {
      kernel &k = *(new kernel(this,
                               filename,
                               kernelName,
                               kernelProps));

      k.binaryFilename = filename;

      k.dlHandle = sys::dlopen(filename);
      k.function = sys::dlsym(k.dlHandle, kernelName);

      return &k;
    }
    //==================================

      //---[ Memory ]-------------------
    memory_v* device::malloc(const udim_t bytes,
                             const void *src,
                             const occa::properties &props) {
      memory *mem = new memory(props);

      mem->dHandle = this;
      mem->size    = bytes;
      mem->ptr     = (char*) sys::malloc(bytes);

      if (src != NULL) {
        ::memcpy(mem->ptr, src, bytes);
      }

      return mem;
    }

    udim_t device::memorySize() const {
      return sys::installedRAM();
    }
    //==================================
  }
}
