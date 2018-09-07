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

#include <occa/core/base.hpp>
#include <occa/tools/env.hpp>
#include <occa/io.hpp>
#include <occa/tools/sys.hpp>
#include <occa/mode/serial/device.hpp>
#include <occa/mode/serial/kernel.hpp>
#include <occa/mode/serial/memory.hpp>
#include <occa/mode/serial/stream.hpp>
#include <occa/mode/serial/streamTag.hpp>
#include <occa/lang/mode/serial.hpp>

namespace occa {
  namespace serial {
    device::device(const occa::properties &properties_) :
      occa::modeDevice_t(properties_) {

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

      if (properties.get<std::string>("kernel/compiler_flags").size()) {
        compilerFlags = (std::string) properties["kernel/compiler_flags"];
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

      if (properties.get<std::string>("kernel/compiler_env_script").size()) {
        compilerEnvScript = (std::string) properties["kernel/compiler_env_script"];
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

        if (visualStudioTools) {
          compilerEnvScript = "\"" + std::string(visualStudioTools) + "..\\..\\VC\\vcvarsall.bat\" " + byteness;
        } else {
          std::cout << "WARNING: Visual Studio environment variable not found -> compiler environment (vcvarsall.bat) maybe not correctly setup." << std::endl;
        }
#endif
      }

      properties["kernel/vendor"] = vendor;
      sys::addSharedBinaryFlagsTo(vendor, compilerFlags);

      properties["kernel/compiler"] = compiler;
      properties["kernel/compiler_flags"] = compilerFlags;
      properties["kernel/compiler_env_script"] = compilerEnvScript;
    }

    device::~device() {}

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

    hash_t device::kernelHash(const occa::properties &props) const {
      return (
        occa::hash(props["vendor"])
        ^ props["compiler"]
        ^ props["compiler_flags"]
        ^ props["compiler_env_script"]
      );
    }

    //---[ Stream ]---------------------
    modeStream_t* device::createStream(const occa::properties &props) {
      return new stream(this, props);
    }

    occa::streamTag device::tagStream() {
      return new occa::serial::streamTag(this, sys::currentTime());
    }

    void device::waitFor(occa::streamTag tag) {}

    double device::timeBetween(const occa::streamTag &startTag,
                               const occa::streamTag &endTag) {
      occa::serial::streamTag *srStartTag = (
        dynamic_cast<occa::serial::streamTag*>(startTag.getModeStreamTag())
      );
      occa::serial::streamTag *srEndTag = (
        dynamic_cast<occa::serial::streamTag*>(endTag.getModeStreamTag())
      );

      return (srEndTag->time - srStartTag->time);
    }
    //==================================

    //---[ Kernel ]---------------------
    bool device::parseFile(const std::string &filename,
                           const std::string &outputFile,
                           const occa::properties &kernelProps,
                           lang::kernelMetadataMap &metadata) {
      lang::okl::serialParser parser(kernelProps);
      parser.parseFile(filename);

      // Verify if parsing succeeded
      if (!parser.succeeded()) {
        OCCA_ERROR("Unable to transform OKL kernel",
                   kernelProps.get("silent", false));
        return false;
      }

      if (!io::isFile(outputFile)) {
        hash_t hash = occa::hash(outputFile);
        io::lock_t lock(hash, "serial-parser");
        if (lock.isMine()) {
          parser.writeToFile(outputFile);
        }
      }

      parser.setMetadata(metadata);

      return true;
    }

    modeKernel_t* device::buildKernel(const std::string &filename,
                                      const std::string &kernelName,
                                      const hash_t kernelHash,
                                      const occa::properties &kernelProps) {
      const std::string hashDir = io::hashDir(filename, kernelHash);
      // Binary name depends if this is being used as the launcher kernel for
      //   GPU-style kernels
      const std::string &kcBinaryFile = (
        (filename != (hashDir + kc::hostSourceFile))
        ? kc::binaryFile
        : kc::hostBinaryFile
      );
      std::string binaryFilename = hashDir + kcBinaryFile;
      bool foundBinary = true;

      // Check if binary exists and is finished
      io::lock_t lock;
      if (!io::cachedFileIsComplete(hashDir, kcBinaryFile) ||
          !io::isFile(binaryFilename)) {
        lock = io::lock_t(kernelHash, "serial-kernel");
        if (lock.isMine()) {
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
        modeKernel_t *k = buildKernelFromBinary(binaryFilename,
                                                kernelName,
                                                kernelProps);
        if (k) {
          k->sourceFilename = filename;
        }
        return k;
      }

      // Cache raw origin
      std::string sourceFilename = (
        io::cacheFile(filename,
                      kc::rawSourceFile,
                      kernelHash,
                      assembleKernelHeader(kernelProps))
      );

      lang::kernelMetadataMap metadata;
      if (kernelProps.get("okl", true)) {
        const std::string outputFile = hashDir + kc::sourceFile;
        bool valid = parseFile(sourceFilename,
                               outputFile,
                               kernelProps,
                               metadata);
        if (!valid) {
          return NULL;
        }
        sourceFilename = outputFile;

        writeKernelBuildFile(hashDir + kc::buildFile,
                             kernelHash,
                             kernelProps,
                             metadata);
      }

      std::stringstream command;
      std::string compilerEnvScript = kernelProps["compiler_env_script"];
      if (compilerEnvScript.size()) {
        command << compilerEnvScript << " && ";
      }

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
      command << (std::string) kernelProps["compiler"]
              << ' '    << (std::string) kernelProps["compiler_flags"]
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
              << ' '       << kernelProps["compiler_flags"]
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

      modeKernel_t *k = buildKernelFromBinary(binaryFilename,
                                              kernelName,
                                              kernelProps);
      if (k) {
        io::markCachedFileComplete(hashDir, kcBinaryFile);
        k->sourceFilename = filename;
      }
      return k;
    }

    modeKernel_t* device::buildKernelFromBinary(const std::string &filename,
                                                const std::string &kernelName,
                                                const occa::properties &kernelProps) {
      kernel &k = *(new kernel(this,
                               kernelName,
                               filename,
                               kernelProps));

      k.binaryFilename = filename;

      k.dlHandle = sys::dlopen(filename);
      k.function = sys::dlsym(k.dlHandle, kernelName);

      return &k;
    }
    //==================================

    //---[ Memory ]-------------------
    modeMemory_t* device::malloc(const udim_t bytes,
                                 const void *src,
                                 const occa::properties &props) {
      memory *mem = new memory(this, bytes, props);

      mem->ptr = (char*) sys::malloc(bytes);
      if (src) {
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
