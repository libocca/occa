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

#include "occa/modes/serial/kernel.hpp"
#include "occa/tools/env.hpp"
#include "occa/tools/io.hpp"
#include "occa/base.hpp"

namespace occa {
  namespace serial {
    kernel::kernel(const occa::properties &properties_) :
      occa::kernel_v(properties_) {
      dlHandle = NULL;
      handle   = NULL;
    }

    kernel::~kernel() {}

    void kernel::build(const std::string &filename,
                       const std::string &kernelName,
                       const hash_t hash,
                       const occa::properties &props) {

      name = kernelName;
      properties += props;

      const bool isLaunchKernel = props.has("defines/OCCA_LAUNCH_KERNEL");

      const std::string sourceFile = (isLaunchKernel
                                      ? getLaunchSourceFilename(filename, hash)
                                      : getSourceFilename(filename, hash));
      const std::string binaryFile = (isLaunchKernel
                                      ? getLaunchBinaryFilename(filename, hash)
                                      : getBinaryFilename(filename, hash));
      const std::string sourceBasename = (isLaunchKernel
                                          ? kc::launchSourceFile
                                          : kc::sourceFile);
      bool foundBinary = true;

      const std::string hashTag = "serial-kernel";
      if (!io::haveHash(hash, hashTag)) {
        io::waitForHash(hash, hashTag);
      } else if (sys::fileExists(binaryFile)) {
        io::releaseHash(hash, hashTag);
      } else {
        foundBinary = false;
      }

      if (foundBinary) {
        if (settings().get("verboseCompilation", true)) {
          std::cout << "Found cached binary of [" << io::shortname(filename) << "] in [" << io::shortname(binaryFile) << "]\n";
        }
        return buildFromBinary(binaryFile, kernelName, props);
      }

      std::string kernelDefines;
      if (properties.has("occa/kernel/defines")) {
        kernelDefines = properties["occa/kernel/defines"].string();
      } else {
        kernelDefines = io::cacheFile(env::OCCA_DIR + "/include/occa/modes/serial/kernelDefines.hpp",
                                      "serialKernelDefines.hpp");
      }

      std::stringstream ss, command;
      ss << "#include \"" << kernelDefines << "\"\n"
         << assembleHeader(properties) << '\n'
         << "#if defined(OCCA_IN_KERNEL) && !OCCA_IN_KERNEL\n"
         << "using namespace occa;\n"
         << "#endif\n";

      const std::string cachedSourceFile = io::cacheFile(filename,
                                                         sourceBasename,
                                                         hash,
                                                         ss.str(),
                                                         properties["footer"].string());

      const std::string &compilerEnvScript = properties["compilerEnvScript"].string();
      if (compilerEnvScript.size()) {
        command << compilerEnvScript << " && ";
      }

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
      command << properties["compiler"].string()
              << ' '    << properties["compilerFlags"].string()
              << ' '    << cachedSourceFile
              << " -o " << binaryFile
              << " -I"  << env::OCCA_DIR << "include"
              << " -L"  << env::OCCA_DIR << "lib -locca"
              << std::endl;
#else
#  if (OCCA_DEBUG_ENABLED)
      const std::string occaLib = env::OCCA_DIR + "lib/libocca_d.lib ";
#  else
      const std::string occaLib = env::OCCA_DIR + "lib/libocca.lib ";
#  endif

      command << properties["compiler"]
              << " /D MC_CL_EXE"
              << " /D OCCA_OS=OCCA_WINDOWS_OS"
              << " /EHsc"
              << " /wd4244 /wd4800 /wd4804 /wd4018"
              << ' '       << properties["compilerFlags"]
              << " /I"     << env::OCCA_DIR << "/include"
              << ' '       << sourceFile
              << " /link " << occaLib
              << " /OUT:"  << binaryFile
              << std::endl;
#endif

      const std::string &sCommand = command.str();

      if (settings().get("verboseCompilation", true)) {
        std::cout << "Compiling [" << kernelName << "]\n" << sCommand << "\n";
      }

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
      const int compileError = system(sCommand.c_str());
#else
      const int compileError = system(("\"" +  sCommand + "\"").c_str());
#endif

      if (compileError) {
        io::releaseHash(hash, hashTag);
        OCCA_ERROR("Compilation error", compileError);
      }

      dlHandle = sys::dlopen(binaryFile, hash, hashTag);
      handle   = sys::dlsym(dlHandle, kernelName, hash, hashTag);

      io::releaseHash(hash, hashTag);
    }

    void kernel::buildFromBinary(const std::string &filename,
                                 const std::string &kernelName,
                                 const occa::properties &props) {

      name = kernelName;
      properties += props;

      dlHandle = sys::dlopen(filename);
      handle   = sys::dlsym(dlHandle, kernelName);
    }

    int kernel::maxDims() const {
      return 3;
    }

    dim kernel::maxOuterDims() const {
      return dim(-1,-1,-1);
    }

    dim kernel::maxInnerDims() const {
      return dim(-1,-1,-1);
    }

    void kernel::runFromArguments(const int kArgc, const kernelArg *kArgs) const {
      int argc = 0;
      kernelInfoArg_t info;

      if (properties.get("OKL", true)) {
        info.outerDim0 = outer.x; info.innerDim0 = inner.x;
        info.outerDim1 = outer.y; info.innerDim1 = inner.y;
        info.outerDim2 = outer.z; info.innerDim2 = inner.z;

        info.innerId0 = info.innerId1 = info.innerId2 = 0;
        vArgs[argc++] = &info;
      }

      for (int i = 0; i < kArgc; ++i) {
        const int argCount = (int) kArgs[i].args.size();
        if (argCount) {
          const kernelArg_t *kArgs_i = &(kArgs[i].args[0]);
          for (int j = 0; j < argCount; ++j) {
            vArgs[argc++] = kArgs_i[j].ptr();
          }
        }
      }

      sys::runFunction(handle, argc, vArgs);
    }

    void kernel::free() {
      if (dlHandle) {
        sys::dlclose(dlHandle);
        dlHandle = NULL;
      }
    }
  }
}
