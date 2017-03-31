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

#if OCCA_CUDA_ENABLED

#include "occa/modes/cuda/kernel.hpp"
#include "occa/modes/cuda/device.hpp"
#include "occa/modes/cuda/utils.hpp"
#include "occa/tools/env.hpp"
#include "occa/tools/io.hpp"
#include "occa/tools/misc.hpp"
#include "occa/tools/sys.hpp"
#include "occa/base.hpp"

namespace occa {
  namespace cuda {
    kernel::kernel(const occa::properties &properties_) :
      occa::kernel_v(properties_) {}

    kernel::~kernel() {}

    void* kernel::getHandle(const occa::properties &props) const {
      const std::string type = props["type"];

      if (type == "kernel") {
        return handle;
      }
      if (type == "module") {
        return module;
      }
      return NULL;
    }

    void kernel::build(const std::string &filename,
                       const std::string &kernelName,
                       const occa::properties &props) {

      occa::properties allProps = properties + props;
      name = kernelName;

      if (allProps.get<std::string>("compilerFlags").find("-arch=sm_") == std::string::npos) {
        const int major = ((cuda::device*) dHandle)->archMajorVersion;
        const int minor = ((cuda::device*) dHandle)->archMinorVersion;
        std::stringstream ss;
        ss << " -arch=sm_" << major << minor << ' ';
        allProps["compilerFlags"].string() += ss.str();
      }

      hash_t hash = occa::hashFile(filename);
      hash ^= allProps.hash();

      const std::string sourceFile    = getSourceFilename(filename, hash);
      const std::string binaryFile    = getBinaryFilename(filename, hash);
      const std::string ptxBinaryFile = io::hashDir(filename, hash) + "ptxBinary.o";
      bool foundBinary = true;

      const std::string hashTag = "cuda-kernel";
      if (!io::haveHash(hash, hashTag)) {
        io::waitForHash(hash, hashTag);
      } else if (sys::fileExists(binaryFile)) {
        io::releaseHash(hash, hashTag);
      } else {
        foundBinary = false;
      }

      if (foundBinary) {
        if (settings().get("verboseCompilation", true)) {
          std::cout << "Found cached binary of [" << io::shortname(filename)
                    << "] in [" << io::shortname(binaryFile) << "]\n";
        }
        return buildFromBinary(binaryFile, kernelName, props);
      }

      const std::string kernelDefines =
        io::cacheFile(env::OCCA_DIR + "/include/occa/modes/cuda/kernelDefines.hpp",
                      "cudaKernelDefines.hpp");

      std::stringstream ss;
      ss << "#include \"" << kernelDefines << "\"\n"
         << assembleHeader(allProps);

      const std::string cachedSourceFile = io::cacheFile(filename,
                                                         kc::sourceFile,
                                                         hash,
                                                         ss.str(),
                                                         allProps["footer"]);

      if (settings().get("verboseCompilation", true)) {
        std::cout << "Compiling [" << kernelName << "]\n";
      }

      //---[ PTX Check Command ]----------
      std::stringstream command;
      if (allProps.has("compilerEnvScript")) {
        command << allProps["compilerEnvScript"] << " && ";
      }

      command << allProps["compiler"]
              << ' '          << allProps["compilerFlags"]
              << " -Xptxas -v,-dlcm=cg"
#if (OCCA_OS == OCCA_WINDOWS_OS)
              << " -D OCCA_OS=OCCA_WINDOWS_OS -D _MSC_VER=1800"
#endif
              << " -I"        << env::OCCA_DIR << "include"
              << " -L"        << env::OCCA_DIR << "lib -locca"
              << " -x cu -c " << cachedSourceFile
              << " -o "       << ptxBinaryFile;

      const std::string &ptxCommand = command.str();
      if (settings().get("verboseCompilation", true)) {
        std::cout << "Compiling [" << kernelName << "]\n" << ptxCommand << "\n";
      }

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
      ignoreResult( system(ptxCommand.c_str()) );
#else
      ignoreResult( system(("\"" +  ptxCommand + "\"").c_str()) );
#endif

      //---[ Compiling Command ]----------
      command.str("");
      command << allProps["compiler"]
              << ' '       << allProps["compilerFlags"]
              << " -ptx"
#if (OCCA_OS == OCCA_WINDOWS_OS)
              << " -D OCCA_OS=OCCA_WINDOWS_OS -D _MSC_VER=1800"
#endif
              << " -I"        << env::OCCA_DIR << "include"
              << " -L"        << env::OCCA_DIR << "lib -locca"
              << " -x cu " << cachedSourceFile
              << " -o "    << binaryFile;

      const std::string &sCommand = command.str();
      if (settings().get("verboseCompilation", true)) {
        std::cout << sCommand << '\n';
      }

      const int compileError = system(sCommand.c_str());

      if (compileError) {
        io::releaseHash(hash, hashTag);
        OCCA_ERROR("Compilation error",
                   false);
      }

      const CUresult moduleLoadError = cuModuleLoad(&module,
                                                    binaryFile.c_str());

      if (moduleLoadError) {
        io::releaseHash(hash, hashTag);
      }

      OCCA_CUDA_ERROR("Kernel (" + name + ") : Loading Module",
                      moduleLoadError);

      const CUresult moduleGetFunctionError = cuModuleGetFunction(&handle,
                                                                  module,
                                                                  name.c_str());

      if (moduleGetFunctionError) {
        io::releaseHash(hash, hashTag);
      }

      OCCA_CUDA_ERROR("Kernel (" + name + ") : Loading Function",
                      moduleGetFunctionError);

      io::releaseHash(hash, hashTag);
    }

    void kernel::buildFromBinary(const std::string &filename,
                                 const std::string &kernelName,
                                 const occa::properties &props) {
      name = kernelName;

      OCCA_CUDA_ERROR("Kernel (" + kernelName + ") : Loading Module",
                      cuModuleLoad(&module, filename.c_str()));

      OCCA_CUDA_ERROR("Kernel (" + kernelName + ") : Loading Function",
                      cuModuleGetFunction(&handle, module, kernelName.c_str()));
    }

    int kernel::maxDims() const {
      return 3;
    }

    dim kernel::maxOuterDims() const {
      return dim(-1, -1, -1);
    }

    dim kernel::maxInnerDims() const {
      static dim innerDims(0);
      if (innerDims.x == 0) {
        int maxSize;
        OCCA_CUDA_ERROR("Kernel: Getting Maximum Inner-Dim Size",
                        cuFuncGetAttribute(&maxSize,
                                           CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                           handle));

        innerDims.x = maxSize;
      }
      return innerDims;
    }

    void kernel::runFromArguments(const int kArgc, const kernelArg *kArgs) const {
      int occaKernelInfoArgs = 0;
      int argc = 0;

      const bool addInfoArgs = properties.get("OKL", true);

      void **vArgs = new void*[addInfoArgs + kernelArg::argumentCount(kArgc, kArgs)];

      if (addInfoArgs) {
        vArgs[argc++] = &occaKernelInfoArgs;
      }

      for (int i = 0; i < kArgc; ++i) {
        const kernelArg_t &arg = kArgs[i].arg;
        const dim_t extraArgCount = kArgs[i].extraArgs.size();
        const kernelArg_t *extraArgs = extraArgCount ? &(kArgs[i].extraArgs[0]) : NULL;

        vArgs[argc++] = arg.ptr();
        for (int j = 0; j < extraArgCount; ++j) {
          vArgs[argc++] = extraArgs[j].ptr();
        }
      }

      OCCA_CUDA_ERROR("Launching Kernel",
                      cuLaunchKernel(handle,
                                     outer.x, outer.y, outer.z,
                                     inner.x, inner.y, inner.z,
                                     0, *((CUstream*) dHandle->currentStream),
                                     vArgs, 0));
      delete [] vArgs;
    }

    void kernel::free() {
      if (module) {
        OCCA_CUDA_ERROR("Kernel (" + name + ") : Unloading Module",
                        cuModuleUnload(module));
        module = NULL;
      }
    }
  }
}

#endif
