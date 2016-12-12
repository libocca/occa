/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2016 David Medina and Tim Warburton
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
#include "occa/base.hpp"

namespace occa {
  namespace cuda {
    kernel::kernel() {
      strMode = "CUDA";

      data    = NULL;
      dHandle = NULL;

      dims  = 1;
      inner = occa::dim(1,1,1);
      outer = occa::dim(1,1,1);

      maximumInnerDimSize_ = 0;
      preferredDimSize_    = 0;
    }

    kernel::kernel(const kernel &k) {
      *this = k;
    }

    kernel& kernel::operator = (const kernel &k) {
      data    = k.data;
      dHandle = k.dHandle;

      metaInfo = k.metaInfo;

      dims  = k.dims;
      inner = k.inner;
      outer = k.outer;

      nestedKernels = k.nestedKernels;

      preferredDimSize_ = k.preferredDimSize_;

      return *this;
    }

    kernel::~kernel() {}

    void* kernel::getKernelHandle() {
      OCCA_EXTRACT_DATA(CUDA, Kernel);

      return data_.function;
    }

    void* kernel::getProgramHandle() {
      OCCA_EXTRACT_DATA(CUDA, Kernel);

      return data_.module;
    }

    kernel* kernel::buildFromSource(const std::string &filename,
                                                    const std::string &functionName,
                                                    const kernelInfo &info_) {

      name = functionName;

      OCCA_EXTRACT_DATA(CUDA, Kernel);
      kernelInfo info = info_;

      dHandle->addOccaHeadersToInfo(info);

      // Add arch to info (for hash purposes)
      if ((dHandle->compilerFlags.find("-arch=sm_") == std::string::npos) &&
         (            info.flags.find("-arch=sm_") == std::string::npos)) {

        std::stringstream ss;
        int major, minor;

        OCCA_CUDA_CHECK("Kernel (" + functionName + ") : Getting CUDA Device Arch",
                        cuDeviceComputeCapability(&major, &minor, data_.device) );

        ss << " -arch=sm_" << major << minor << ' ';
        info.flags += ss.str();
      }

      const std::string hash = getFileContentHash(filename,
                                                  dHandle->getInfoSalt(info));

      const std::string hashDir = hashDir(filename, hash);
      const std::string hashTag = "cuda-kernel";

      const std::string sourceFile    = hashDir + kc::sourceFile;
      const std::string binaryFile    = hashDir + binaryName(kc::binaryFile);
      const std::string ptxBinaryFile = hashDir + "ptxBinary.o";
      bool foundBinary = true;

      if (!haveHash(hash, hashTag)) {
        waitForHash(hash, hashTag);
      } else if (sys::fileExists(binaryFile)) {
        releaseHash(hash, hashTag);
      } else {
        foundBinary = false;
      }

      if (foundBinary) {
        if (verboseCompilation_f) {
          std::cout << "Found cached binary of [" << io::shortname(filename) << "] in [" << io::shortname(binaryFile) << "]\n";
        }
        return buildFromBinary(binaryFile, functionName);
      }

      createSourceFileFrom(filename, hashDir, info);

      std::stringstream command;

      if (verboseCompilation_f) {
        std::cout << "Compiling [" << functionName << "]\n";
      }

#if 0
      //---[ PTX Check Command ]----------
      if (dHandle->compilerEnvScript.size()) {
        command << dHandle->compilerEnvScript << " && ";
      }

      command << dHandle->compiler
              << " -I."
              << " -I"  << env::OCCA_DIR << "include"
#  if (OCCA_OS == OCCA_WINDOWS_OS)
              << " -D OCCA_OS=OCCA_WINDOWS_OS -D _MSC_VER=1800"
#  endif
              << ' '          << dHandle->compilerFlags
              << " -Xptxas -v,-dlcm=cg"
              << ' '          << info.flags
              << " -x cu -c " << sourceFile
              << " -o "       << ptxBinaryFile;

      const std::string &ptxCommand = command.str();

      if (verboseCompilation_f) {
        std::cout << "Compiling [" << functionName << "]\n" << ptxCommand << "\n";
      }

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
      ignoreResult( system(ptxCommand.c_str()) );
#else
      ignoreResult( system(("\"" +  ptxCommand + "\"").c_str()) );
#endif
#endif

      //---[ Compiling Command ]----------
      command.str("");

      command << dHandle->compiler
              << " -o "       << binaryFile
              << " -ptx -I."
              << " -I"  << env::OCCA_DIR << "include"
#  if (OCCA_OS == OCCA_WINDOWS_OS)
              << " -D OCCA_OS=OCCA_WINDOWS_OS -D _MSC_VER=1800"
#  endif
              << ' '          << dHandle->compilerFlags
              << archSM
              << ' '          << info.flags
              << " -x cu "    << sourceFile;

      const std::string &sCommand = command.str();

      if (verboseCompilation_f) {
        std::cout << sCommand << '\n';
      }

      const int compileError = system(sCommand.c_str());

      if (compileError) {
        releaseHash(hash, hashTag);
        OCCA_CHECK(false, "Compilation error");
      }

      const CUresult moduleLoadError = cuModuleLoad(&data_.module,
                                                    binaryFile.c_str());

      if (moduleLoadError) {
        releaseHash(hash, hashTag);
      }

      OCCA_CUDA_CHECK("Kernel (" + functionName + ") : Loading Module",
                      moduleLoadError);

      const CUresult moduleGetFunctionError = cuModuleGetFunction(&data_.function,
                                                                  data_.module,
                                                                  functionName.c_str());

      if (moduleGetFunctionError) {
        releaseHash(hash, hashTag);
      }

      OCCA_CUDA_CHECK("Kernel (" + functionName + ") : Loading Function",
                      moduleGetFunctionError);

      releaseHash(hash, hashTag);

      return this;
    }

    kernel* kernel::buildFromBinary(const std::string &filename,
                                                    const std::string &functionName) {

      name = functionName;

      OCCA_EXTRACT_DATA(CUDA, Kernel);

      OCCA_CUDA_CHECK("Kernel (" + functionName + ") : Loading Module",
                      cuModuleLoad(&data_.module, filename.c_str()));

      OCCA_CUDA_CHECK("Kernel (" + functionName + ") : Loading Function",
                      cuModuleGetFunction(&data_.function, data_.module, functionName.c_str()));

      return this;
    }

    kernel* kernel::loadFromLibrary(const char *cache,
                                                    const std::string &functionName) {
      OCCA_EXTRACT_DATA(CUDA, Kernel);

      OCCA_CUDA_CHECK("Kernel (" + functionName + ") : Loading Module",
                      cuModuleLoadData(&data_.module, cache));

      OCCA_CUDA_CHECK("Kernel (" + functionName + ") : Loading Function",
                      cuModuleGetFunction(&data_.function, data_.module, functionName.c_str()));

      return this;
    }

    udim_t kernel::maximumInnerDimSize() {
      if (maximumInnerDimSize_)
        return maximumInnerDimSize_;

      OCCA_EXTRACT_DATA(CUDA, Kernel);

      int maxSize;

      OCCA_CUDA_CHECK("Kernel: Getting Maximum Inner-Dim Size",
                      cuFuncGetAttribute(&maxSize, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, data_.function));

      maximumInnerDimSize_ = (udim_t) maxSize;

      return maximumInnerDimSize_;
    }

    int kernel::preferredDimSize() {
      preferredDimSize_ = 32;
      return 32;
    }

    void kernel::runFromArguments(const int kArgc, const kernelArg *kArgs) {
      CUDAKernelData_t &data_ = *((CUDAKernelData_t*) data);
      CUfunction function_ = data_.function;

      int occaKernelInfoArgs = 0;
      int argc = 0;

      data_.vArgs = new void*[1 + kernelArg::argumentCount(kArgc, kArgs)];
      data_.vArgs[argc++] = &occaKernelInfoArgs;
      for (int i = 0; i < kArgc; ++i) {
        for (int j = 0; j < kArgs[i].argc; ++j) {
          data_.vArgs[argc++] = kArgs[i].args[j].ptr();
        }
      }

      OCCA_CUDA_CHECK("Launching Kernel",
                      cuLaunchKernel(function_,
                                     outer.x, outer.y, outer.z,
                                     inner.x, inner.y, inner.z,
                                     0, *((CUstream*) dHandle->currentStream),
                                     data_.vArgs, 0));
      delete [] data_.vArgs;
    }

    void kernel::free() {
      OCCA_EXTRACT_DATA(CUDA, Kernel);

      OCCA_CUDA_CHECK("Kernel (" + name + ") : Unloading Module",
                      cuModuleUnload(data_.module));

      delete (CUDAKernelData_t*) this->data;
    }
  }
}

#endif
