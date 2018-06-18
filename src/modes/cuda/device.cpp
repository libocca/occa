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

#if OCCA_CUDA_ENABLED

#include <occa/base.hpp>
#include <occa/tools/env.hpp>
#include <occa/tools/misc.hpp>
#include <occa/tools/sys.hpp>
#include <occa/modes/cuda/device.hpp>
#include <occa/modes/cuda/kernel.hpp>
#include <occa/modes/cuda/memory.hpp>
#include <occa/modes/cuda/utils.hpp>
#include <occa/lang/kernelMetadata.hpp>
#include <occa/lang/primitive.hpp>
#include <occa/lang/modes/cuda.hpp>

namespace occa {
  namespace cuda {
    device::device(const occa::properties &properties_) :
      occa::device_v(properties_) {

      if (!properties.has("wrapped")) {
        OCCA_ERROR("[CUDA] device not given a [deviceID] integer",
                   properties.has("deviceID") &&
                   properties["deviceID"].isNumber());

        const int deviceID = properties.get<int>("deviceID");

        OCCA_CUDA_ERROR("Device: Creating Device",
                        cuDeviceGet(&cuDevice, deviceID));

        OCCA_CUDA_ERROR("Device: Creating Context",
                        cuCtxCreate(&cuContext, CU_CTX_SCHED_AUTO, cuDevice));
      }

      p2pEnabled = false;

      std::string compiler = properties["kernel/compiler"];
      std::string compilerFlags = properties["kernel/compilerFlags"];

      if (!compiler.size()) {
        if (env::var("OCCA_CUDA_COMPILER").size()) {
          compiler = env::var("OCCA_CUDA_COMPILER");
        } else {
          compiler = "nvcc";
        }
      }

      if (!compilerFlags.size()) {
        compilerFlags = env::var("OCCA_CUDA_COMPILER_FLAGS");
      }

      properties["kernel/compiler"]      = compiler;
      properties["kernel/compilerFlags"] = compilerFlags;

      OCCA_CUDA_ERROR("Device: Getting CUDA Device Arch",
                      cuDeviceComputeCapability(&archMajorVersion,
                                                &archMinorVersion,
                                                cuDevice) );

      archMajorVersion = properties.get("cuda/arch/major", archMajorVersion);
      archMinorVersion = properties.get("cuda/arch/minor", archMinorVersion);

      properties["kernel/verbose"] = properties.get("verbose", false);
    }

    device::~device() {}

    void device::free() {
      if (cuContext) {
        OCCA_CUDA_ERROR("Device: Freeing Context",
                        cuCtxDestroy(cuContext) );
        cuContext = NULL;
      }
    }

    void device::finish() const {
      OCCA_CUDA_ERROR("Device: Finish",
                      cuStreamSynchronize(*((CUstream*) currentStream)) );
    }

    bool device::hasSeparateMemorySpace() const {
      return true;
    }

    hash_t device::hash() const {
      if (!hash_.initialized) {
        std::stringstream ss;
        ss << "major: " << archMajorVersion << ' '
           << "minor: " << archMinorVersion;
        hash_ = occa::hash(ss.str());
      }
      return hash_;
    }

    //---[ Stream ]---------------------
    stream_t device::createStream() const {
      CUstream *retStream = new CUstream;

      OCCA_CUDA_ERROR("Device: Setting Context",
                      cuCtxSetCurrent(cuContext));
      OCCA_CUDA_ERROR("Device: createStream",
                      cuStreamCreate(retStream, CU_STREAM_DEFAULT));

      return retStream;
    }

    void device::freeStream(stream_t s) const {
      OCCA_CUDA_ERROR("Device: freeStream",
                      cuStreamDestroy( *((CUstream*) s) ));
      delete (CUstream*) s;
    }

    streamTag device::tagStream() const {
      streamTag ret;

      OCCA_CUDA_ERROR("Device: Setting Context",
                      cuCtxSetCurrent(cuContext));
      OCCA_CUDA_ERROR("Device: Tagging Stream (Creating Tag)",
                      cuEventCreate(&cuda::event(ret), CU_EVENT_DEFAULT));
      OCCA_CUDA_ERROR("Device: Tagging Stream",
                      cuEventRecord(cuda::event(ret), 0));

      return ret;
    }

    void device::waitFor(streamTag tag) const {
      OCCA_CUDA_ERROR("Device: Waiting For Tag",
                      cuEventSynchronize(cuda::event(tag)));
    }

    double device::timeBetween(const streamTag &startTag,
                               const streamTag &endTag) const {
      OCCA_CUDA_ERROR("Device: Waiting for endTag",
                      cuEventSynchronize(cuda::event(endTag)));

      float msTimeTaken;
      OCCA_CUDA_ERROR("Device: Timing Between Tags",
                      cuEventElapsedTime(&msTimeTaken, cuda::event(startTag), cuda::event(endTag)));

      return (double) (1.0e-3 * (double) msTimeTaken);
    }

    stream_t device::wrapStream(void *handle_,
                                const occa::properties &props) const {
      return handle_;
    }
    //==================================

    //---[ Kernel ]---------------------
    bool device::parseFile(const std::string &filename,
                           const std::string &outputFile,
                           const std::string &hostOutputFile,
                           const occa::properties &kernelProps,
                           lang::kernelMetadataMap &hostMetadata,
                           lang::kernelMetadataMap &deviceMetadata) {
      lang::okl::cudaParser parser(kernelProps);
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
        io::lock_t lock(hash, "cuda-parser-device");
        if (lock.isMine()) {
          parser.writeToFile(outputFile);
        }
      }

      if (!sys::fileExists(hostOutputFile)) {
        hash_t hash = occa::hash(hostOutputFile);
        io::lock_t lock(hash, "cuda-parser-host");
        if (lock.isMine()) {
          parser.hostParser.writeToFile(hostOutputFile);
        }
      }

      parser.hostParser.setMetadata(hostMetadata);
      parser.setMetadata(deviceMetadata);

      return true;
    }

    kernel_v* device::buildKernel(const std::string &filename,
                                  const std::string &kernelName,
                                  const hash_t kernelHash,
                                  const occa::properties &kernelProps) {

      occa::properties allProps = properties["kernel"] + kernelProps;

      const std::string hashDir = io::hashDir(filename, kernelHash);
      const std::string binaryFilename = hashDir + kc::binaryFile;
      bool foundBinary = true;
      bool usingOKL = allProps.get("okl", true);

      io::lock_t lock(kernelHash, "cuda-kernel");
      if (lock.isMine()) {
        if (sys::fileExists(binaryFilename)) {
          lock.release();
        } else {
          foundBinary = false;
        }
      }

      const bool verbose = allProps.get("verbose", false);
      if (foundBinary) {
        if (verbose) {
           std::cout << "Loading cached ["
                     << kernelName
                     << "] from ["
                     << io::shortname(filename)
                     << "] in [" << io::shortname(binaryFilename) << "]\n";
        }
        if (usingOKL) {
          lang::kernelMetadataMap hostMetadata = (
            lang::getBuildFileMetadata(hashDir + kc::hostBuildFile)
          );
          lang::kernelMetadataMap deviceMetadata = (
            lang::getBuildFileMetadata(hashDir + kc::buildFile)
          );
          return buildOKLKernelFromBinary(hashDir,
                                          kernelName,
                                          hostMetadata,
                                          deviceMetadata,
                                          allProps,
                                          lock);
        } else {
          return buildKernelFromBinary(binaryFilename,
                                       kernelName,
                                       kernelProps);
        }
      }

      // Cache raw origin
      std::string sourceFilename = (
        io::cacheFile(filename,
                      kc::rawSourceFile,
                      kernelHash,
                      assembleHeader(allProps))
      );

      kernel_v *launcherKernel = NULL;
      lang::kernelMetadataMap hostMetadata, deviceMetadata;
      if (usingOKL) {
        const std::string outputFile = hashDir + kc::sourceFile;
        const std::string hostOutputFile = hashDir + kc::hostSourceFile;
        bool valid = parseFile(sourceFilename,
                               outputFile,
                               hostOutputFile,
                               allProps,
                               hostMetadata,
                               deviceMetadata);
        if (!valid) {
          return NULL;
        }
        sourceFilename = outputFile;

        launcherKernel = buildLauncherKernel(hashDir,
                                             kernelName,
                                             hostMetadata[kernelName]);

        // No OKL means no build file is generated,
        //   so we need to build it
        host()
          .getDHandle()
          ->writeKernelBuildFile(hashDir + kc::hostBuildFile,
                                 kernelHash,
                                 occa::properties(),
                                 hostMetadata);

        writeKernelBuildFile(hashDir + kc::buildFile,
                             kernelHash,
                             allProps,
                             deviceMetadata);
      }

      compileKernel(hashDir,
                    kernelName,
                    allProps,
                    lock);

      // Regular CUDA Kernel
      if (!launcherKernel) {
        CUmodule cuModule;
        CUfunction cuFunction;
        CUresult error;

        error = cuModuleLoad(&cuModule, binaryFilename.c_str());
        if (error) {
          lock.release();
          OCCA_CUDA_ERROR("Kernel [" + kernelName + "]: Loading Module",
                          error);
        }
        error = cuModuleGetFunction(&cuFunction,
                                    cuModule,
                                    kernelName.c_str());
        if (error) {
          lock.release();
          OCCA_CUDA_ERROR("Kernel [" + kernelName + "]: Loading Function",
                          error);
        }
        return new kernel(this,
                          kernelName,
                          sourceFilename,
                          cuModule,
                          cuFunction,
                          allProps);
      }

      return buildOKLKernelFromBinary(hashDir,
                                      kernelName,
                                      hostMetadata,
                                      deviceMetadata,
                                      allProps,
                                      lock);
    }

    void device::setArchCompilerFlags(occa::properties &kernelProps) {
      if (kernelProps.get<std::string>("compilerFlags").find("-arch=sm_") == std::string::npos) {
        const int major = archMajorVersion;
        const int minor = archMinorVersion;
        std::stringstream ss;
        ss << " -arch=sm_" << major << minor << ' ';
        kernelProps["compilerFlags"] += ss.str();
      }
    }

    void device::compileKernel(const std::string &hashDir,
                               const std::string &kernelName,
                               occa::properties &kernelProps,
                               io::lock_t &lock) {

      const bool verbose = kernelProps.get("verbose", false);

      std::string sourceFilename = hashDir + kc::sourceFile;
      std::string binaryFilename = hashDir + kc::binaryFile;
      const std::string ptxBinaryFilename = hashDir + "ptx_binary.o";

      setArchCompilerFlags(kernelProps);

      //---[ PTX Check Command ]--------
      std::stringstream command;
      if (kernelProps.has("compilerEnvScript")) {
        command << kernelProps["compilerEnvScript"] << " && ";
      }

      command << kernelProps["compiler"]
              << ' ' << kernelProps["compilerFlags"]
              << " -Xptxas -v,-dlcm=cg"
#if (OCCA_OS == OCCA_WINDOWS_OS)
              << " -D OCCA_OS=OCCA_WINDOWS_OS -D _MSC_VER=1800"
#endif
              << " -I"        << env::OCCA_DIR << "include"
              << " -L"        << env::OCCA_DIR << "lib -locca"
              << " -x cu -c " << sourceFilename
              << " -o "       << ptxBinaryFilename;

      if (!verbose) {
        command << " > /dev/null 2>&1";
      }
      const std::string &ptxCommand = command.str();
      if (verbose) {
        std::cout << "Compiling [" << kernelName << "]\n" << ptxCommand << "\n";
      }

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
      ignoreResult( system(ptxCommand.c_str()) );
#else
      ignoreResult( system(("\"" +  ptxCommand + "\"").c_str()) );
#endif
      //================================

      //---[ Compiling Command ]--------
      command.str("");
      command << kernelProps["compiler"]
              << ' ' << kernelProps["compilerFlags"]
              << " -ptx"
#if (OCCA_OS == OCCA_WINDOWS_OS)
              << " -D OCCA_OS=OCCA_WINDOWS_OS -D _MSC_VER=1800"
#endif
              << " -I"        << env::OCCA_DIR << "include"
              << " -L"        << env::OCCA_DIR << "lib -locca"
              << " -x cu " << sourceFilename
              << " -o "    << binaryFilename;

      if (!verbose) {
        command << " > /dev/null 2>&1";
      }
      const std::string &sCommand = command.str();
      if (verbose) {
        std::cout << sCommand << '\n';
      }

      const int compileError = system(sCommand.c_str());

      lock.release();
      if (compileError) {
        OCCA_FORCE_ERROR("Error compiling [" << kernelName << "],"
                         " Command: [" << sCommand << ']');
      }
      //================================
    }

    kernel_v* device::buildOKLKernelFromBinary(const std::string &hashDir,
                                               const std::string &kernelName,
                                               lang::kernelMetadataMap &hostMetadata,
                                               lang::kernelMetadataMap &deviceMetadata,
                                               const occa::properties &kernelProps,
                                               const io::lock_t &lock) {

      const std::string sourceFilename = hashDir + kc::sourceFile;
      const std::string binaryFilename = hashDir + kc::binaryFile;

      CUmodule cuModule;
      CUresult error;

      error = cuModuleLoad(&cuModule, binaryFilename.c_str());
      if (error) {
        lock.release();
        OCCA_CUDA_ERROR("Kernel [" + kernelName + "]: Loading Module",
                        error);
      }

      // Create wrapper kernel and set launcherKernel
      kernel &k = *(new kernel(this,
                               kernelName,
                               sourceFilename,
                               kernelProps));

      k.launcherKernel = buildLauncherKernel(hashDir,
                                             kernelName,
                                             hostMetadata[kernelName]);

      // Find clKernels
      typedef std::map<int, lang::kernelMetadata> kernelOrderMap;
      kernelOrderMap cuKernelMetadata;

      const std::string prefix = "_occa_" + kernelName + "_";

      lang::kernelMetadataMap::iterator it = deviceMetadata.begin();
      while (it != deviceMetadata.end()) {
        const std::string &name = it->first;
        lang::kernelMetadata &metadata = it->second;
        ++it;
        if (!startsWith(name, prefix)) {
          continue;
        }
        std::string suffix = name.substr(prefix.size());
        const char *c = suffix.c_str();
        primitive number = primitive::load(c, false);
        // Make sure we reached the end ['\0']
        //   and have a number
        if (*c || number.isNaN()) {
          continue;
        }
        cuKernelMetadata[number] = metadata;
      }

      kernelOrderMap::iterator oit = cuKernelMetadata.begin();
      while (oit != cuKernelMetadata.end()) {
        lang::kernelMetadata &metadata = oit->second;

        CUfunction cuFunction;
        error = cuModuleGetFunction(&cuFunction,
                                    cuModule,
                                    metadata.name.c_str());
        if (error) {
          lock.release();
          OCCA_CUDA_ERROR("Kernel [" + metadata.name + "]: Loading Function",
                          error);
        }

        kernel *cuKernel = new kernel(this,
                                      metadata.name,
                                      sourceFilename,
                                      cuModule,
                                      cuFunction,
                                      kernelProps);
        cuKernel->dontUseRefs();
        k.cuKernels.push_back(cuKernel);

        ++oit;
      }

      return &k;
    }

    kernel_v* device::buildLauncherKernel(const std::string &hashDir,
                                          const std::string &kernelName,
                                          lang::kernelMetadata &hostMetadata) {
      const std::string hostOutputFile = hashDir + kc::hostSourceFile;

      occa::kernel hostKernel = host().buildKernel(hostOutputFile,
                                                   kernelName,
                                                   "okl: false");

      // Launcher and clKernels use the same refs as the wrapper kernel
      kernel_v *launcherKernel = hostKernel.getKHandle();
      launcherKernel->dontUseRefs();

      launcherKernel->metadata = hostMetadata;

      return launcherKernel;
    }

    kernel_v* device::buildKernelFromBinary(const std::string &filename,
                                            const std::string &kernelName,
                                            const occa::properties &kernelProps) {

      occa::properties allProps = properties["kernel"] + kernelProps;

      CUmodule cuModule;
      CUfunction cuFunction;

      OCCA_CUDA_ERROR("Kernel [" + kernelName + "]: Loading Module",
                      cuModuleLoad(&cuModule, filename.c_str()));

      OCCA_CUDA_ERROR("Kernel [" + kernelName + "]: Loading Function",
                      cuModuleGetFunction(&cuFunction, cuModule, kernelName.c_str()));

      return new kernel(this,
                        kernelName,
                        filename,
                        cuModule,
                        cuFunction,
                        allProps);
    }
    //==================================

    //---[ Memory ]---------------------
    memory_v* device::malloc(const udim_t bytes,
                             const void *src,
                             const occa::properties &props) {

      if (props.get("cuda/mapped", false)) {
        return mappedAlloc(bytes, src, props);
      } else if (props.get("cuda/managed", false)) {
        return managedAlloc(bytes, src, props);
      }

      cuda::memory &mem = *(new cuda::memory(props));
      mem.dHandle = this;
      mem.size    = bytes;

      OCCA_CUDA_ERROR("Device: Setting Context",
                      cuCtxSetCurrent(cuContext));

      OCCA_CUDA_ERROR("Device: malloc",
                      cuMemAlloc(&(mem.cuPtr), bytes));

      if (src != NULL) {
        mem.copyFrom(src, bytes, 0);
      }
      return &mem;
    }

    memory_v* device::mappedAlloc(const udim_t bytes,
                                  const void *src,
                                  const occa::properties &props) {

      cuda::memory &mem = *(new cuda::memory(props));
      mem.dHandle = this;
      mem.size    = bytes;

      OCCA_CUDA_ERROR("Device: Setting Context",
                      cuCtxSetCurrent(cuContext));
      OCCA_CUDA_ERROR("Device: malloc host",
                      cuMemAllocHost((void**) &(mem.mappedPtr), bytes));
      OCCA_CUDA_ERROR("Device: get device pointer from host",
                      cuMemHostGetDevicePointer(&(mem.cuPtr),
                                                mem.mappedPtr,
                                                0));

      if (src != NULL) {
        ::memcpy(mem.mappedPtr, src, bytes);
      }
      return &mem;
    }

    memory_v* device::managedAlloc(const udim_t bytes,
                                   const void *src,
                                   const occa::properties &props) {
      cuda::memory &mem = *(new cuda::memory(props));
#if CUDA_VERSION >= 8000
      mem.dHandle   = this;
      mem.size      = bytes;
      mem.isManaged = true;

      const unsigned int flags = (props.get("cuda/attachedHost", false) ?
                                  CU_MEM_ATTACH_HOST : CU_MEM_ATTACH_GLOBAL);

      OCCA_CUDA_ERROR("Device: Setting Context",
                      cuCtxSetCurrent(cuContext));
      OCCA_CUDA_ERROR("Device: managed alloc",
                      cuMemAllocManaged(&(mem.cuPtr),
                                        bytes,
                                        flags));

      if (src != NULL) {
        mem.copyFrom(src, bytes, 0);
      }
#else
      OCCA_FORCE_ERROR("CUDA version ["
                       << cuda::getVersion()
                       << "] does not support unified memory allocation");
#endif
      return &mem;
    }

    udim_t device::memorySize() const {
      return cuda::getDeviceMemorySize(cuDevice);
    }
    //==================================
  }
}

#endif
