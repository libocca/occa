#include <occa/core/base.hpp>
#include <occa/tools/env.hpp>
#include <occa/tools/misc.hpp>
#include <occa/tools/sys.hpp>
#include <occa/modes/cuda/device.hpp>
#include <occa/modes/cuda/kernel.hpp>
#include <occa/modes/cuda/memory.hpp>
#include <occa/modes/cuda/stream.hpp>
#include <occa/modes/cuda/streamTag.hpp>
#include <occa/modes/cuda/utils.hpp>
#include <occa/modes/serial/device.hpp>
#include <occa/modes/serial/kernel.hpp>
#include <occa/lang/primitive.hpp>
#include <occa/lang/modes/cuda.hpp>

namespace occa {
  namespace cuda {
    device::device(const occa::properties &properties_) :
        occa::launchedModeDevice_t(properties_),
        nullPtr(NULL) {

      if (!properties.has("wrapped")) {
        OCCA_ERROR("[CUDA] device not given a [device_id] integer",
                   properties.has("device_id") &&
                   properties["device_id"].isNumber());

        const int deviceID = properties.get<int>("device_id");

        OCCA_CUDA_ERROR("Device: Creating Device",
                        cuDeviceGet(&cuDevice, deviceID));

        OCCA_CUDA_ERROR("Device: Creating Context",
                        cuCtxCreate(&cuContext, CU_CTX_SCHED_AUTO, cuDevice));
      }

      p2pEnabled = false;

      occa::json &kernelProps = properties["kernel"];
      std::string compiler, compilerFlags;

      if (env::var("OCCA_CUDA_COMPILER").size()) {
        compiler = env::var("OCCA_CUDA_COMPILER");
      } else if (kernelProps.get<std::string>("compiler").size()) {
        compiler = (std::string) kernelProps["compiler"];
      } else {
        compiler = "nvcc";
      }

      if (env::var("OCCA_CUDA_COMPILER_FLAGS").size()) {
        compilerFlags = env::var("OCCA_CUDA_COMPILER_FLAGS");
      } else if (kernelProps.get<std::string>("compiler_flags").size()) {
        compilerFlags = (std::string) kernelProps["compiler_flags"];
      } else {
        compilerFlags = "-O3";
      }

      kernelProps["compiler"] = compiler;
      kernelProps["compiler_flags"] = compilerFlags;

#if CUDA_VERSION < 5000
      OCCA_CUDA_ERROR("Device: Getting CUDA device arch",
                      cuDeviceComputeCapability(&archMajorVersion,
                                                &archMinorVersion,
                                                cuDevice));
#else
      OCCA_CUDA_ERROR("Device: Getting CUDA device major version",
                      cuDeviceGetAttribute(&archMajorVersion,
                                           CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                                           cuDevice));
      OCCA_CUDA_ERROR("Device: Getting CUDA device minor version",
                      cuDeviceGetAttribute(&archMinorVersion,
                                           CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                                           cuDevice));
#endif

      archMajorVersion = kernelProps.get("arch/major", archMajorVersion);
      archMinorVersion = kernelProps.get("arch/minor", archMinorVersion);
    }

    device::~device() {
      if (cuContext) {
        OCCA_CUDA_DESTRUCTOR_ERROR(
          "Device: Freeing Context",
          cuCtxDestroy(cuContext)
        );
        cuContext = NULL;
      }
    }

    void device::finish() const {
      OCCA_CUDA_ERROR("Device: Finish",
                      cuStreamSynchronize(getCuStream()));
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

    hash_t device::kernelHash(const occa::properties &props) const {
      return (
        occa::hash(props["compiler"])
        ^ props["compiler_flags"]
        ^ props["compiler_env_script"]
      );
    }

    lang::okl::withLauncher* device::createParser(const occa::properties &props) const {
      return new lang::okl::cudaParser(props);
    }

    void* device::getNullPtr() {
      if (!nullPtr) {
        // Auto freed through ring garbage collection
        nullPtr = (cuda::memory*) malloc(1, NULL, occa::properties());
      }
      return (void*) &(nullPtr->cuPtr);
    }

    //---[ Stream ]---------------------
    modeStream_t* device::createStream(const occa::properties &props) {
      CUstream cuStream = NULL;

      OCCA_CUDA_ERROR("Device: Setting Context",
                      cuCtxSetCurrent(cuContext));
      OCCA_CUDA_ERROR("Device: createStream",
                      cuStreamCreate(&cuStream, CU_STREAM_DEFAULT));

      return new stream(this, props, cuStream);
    }

    occa::streamTag device::tagStream() {
      CUevent cuEvent = NULL;

      OCCA_CUDA_ERROR("Device: Setting Context",
                      cuCtxSetCurrent(cuContext));
      OCCA_CUDA_ERROR("Device: Tagging Stream (Creating Tag)",
                      cuEventCreate(&cuEvent,
                                    CU_EVENT_DEFAULT));
      OCCA_CUDA_ERROR("Device: Tagging Stream",
                      cuEventRecord(cuEvent, getCuStream()));

      return new occa::cuda::streamTag(this, cuEvent);
    }

    void device::waitFor(occa::streamTag tag) {
      occa::cuda::streamTag *cuTag = (
        dynamic_cast<occa::cuda::streamTag*>(tag.getModeStreamTag())
      );
      OCCA_CUDA_ERROR("Device: Waiting For Tag",
                      cuEventSynchronize(cuTag->cuEvent));
    }

    double device::timeBetween(const occa::streamTag &startTag,
                               const occa::streamTag &endTag) {
      occa::cuda::streamTag *cuStartTag = (
        dynamic_cast<occa::cuda::streamTag*>(startTag.getModeStreamTag())
      );
      occa::cuda::streamTag *cuEndTag = (
        dynamic_cast<occa::cuda::streamTag*>(endTag.getModeStreamTag())
      );

      waitFor(endTag);

      float msTimeTaken;
      OCCA_CUDA_ERROR("Device: Timing Between Tags",
                      cuEventElapsedTime(&msTimeTaken,
                                         cuStartTag->cuEvent,
                                         cuEndTag->cuEvent));

      return (double) (1.0e-3 * (double) msTimeTaken);
    }

    CUstream& device::getCuStream() const {
      occa::cuda::stream *stream = (occa::cuda::stream*) currentStream.getModeStream();
      return stream->cuStream;
    }
    //==================================

    //---[ Kernel ]---------------------
    modeKernel_t* device::buildKernelFromProcessedSource(
      const hash_t kernelHash,
      const std::string &hashDir,
      const std::string &kernelName,
      const std::string &sourceFilename,
      const std::string &binaryFilename,
      const bool usingOkl,
      lang::sourceMetadata_t &launcherMetadata,
      lang::sourceMetadata_t &deviceMetadata,
      const occa::properties &kernelProps,
      io::lock_t lock
    ) {
      compileKernel(hashDir,
                    kernelName,
                    kernelProps,
                    lock);

      if (usingOkl) {
        return buildOKLKernelFromBinary(kernelHash,
                                        hashDir,
                                        kernelName,
                                        launcherMetadata,
                                        deviceMetadata,
                                        kernelProps,
                                        lock);
      }

      // Regular CUDA Kernel
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
                        kernelProps);
    }

    void device::setArchCompilerFlags(const occa::properties &kernelProps,
                                      std::string &compilerFlags) {
      if (compilerFlags.find("-arch=sm_") == std::string::npos) {
        compilerFlags += " -arch=sm_";
        compilerFlags += std::to_string(archMajorVersion);
        compilerFlags += std::to_string(archMinorVersion);
      }
    }

    void device::compileKernel(const std::string &hashDir,
                               const std::string &kernelName,
                               const occa::properties &kernelProps,
                               io::lock_t &lock) {

      occa::properties allProps = kernelProps;
      const bool verbose = allProps.get("verbose", false);

      std::string sourceFilename = hashDir + kc::sourceFile;
      std::string binaryFilename = hashDir + kc::binaryFile;
      const std::string ptxBinaryFilename = hashDir + "ptx_binary.o";

      const std::string compiler = allProps["compiler"];
      std::string compilerFlags = allProps["compiler_flags"];
      const bool compilingOkl = allProps.get("okl/enabled", true);

      setArchCompilerFlags(allProps, compilerFlags);

      if (!compilingOkl) {
        sys::addCompilerIncludeFlags(compilerFlags);
        sys::addCompilerLibraryFlags(compilerFlags);
      }

      //---[ PTX Check Command ]--------
      std::stringstream command;
      if (allProps.has("compiler_env_script")) {
        command << allProps["compiler_env_script"] << " && ";
      }

      command << compiler
              << ' ' << compilerFlags
              << " -Xptxas -v,-dlcm=cg"
#if (OCCA_OS == OCCA_WINDOWS_OS)
              << " -D OCCA_OS=OCCA_WINDOWS_OS -D _MSC_VER=1800"
#endif
              << " -I"        << env::OCCA_DIR << "include"
              << " -I"        << env::OCCA_INSTALL_DIR << "include"
              << " -L"        << env::OCCA_INSTALL_DIR << "lib -locca"
              << " -x cu -c " << sourceFilename
              << " -o "       << ptxBinaryFilename;

      if (!verbose) {
        command << " > /dev/null 2>&1";
      }
      const std::string &ptxCommand = command.str();
      if (verbose) {
        io::stdout << "Compiling [" << kernelName << "]\n" << ptxCommand << "\n";
      }

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
      ignoreResult( system(ptxCommand.c_str()) );
#else
      ignoreResult( system(("\"" +  ptxCommand + "\"").c_str()) );
#endif
      //================================

      //---[ Compiling Command ]--------
      command.str("");
      command << allProps["compiler"]
              << ' ' << compilerFlags
              << " -ptx"
#if (OCCA_OS == OCCA_WINDOWS_OS)
              << " -D OCCA_OS=OCCA_WINDOWS_OS -D _MSC_VER=1800"
#endif
              << " -I"        << env::OCCA_DIR << "include"
              << " -I"        << env::OCCA_INSTALL_DIR << "include"
              << " -L"        << env::OCCA_INSTALL_DIR << "lib -locca"
              << " -x cu " << sourceFilename
              << " -o "    << binaryFilename;

      if (!verbose) {
        command << " > /dev/null 2>&1";
      }
      const std::string &sCommand = command.str();
      if (verbose) {
        io::stdout << sCommand << '\n';
      }

      const int compileError = system(sCommand.c_str());

      lock.release();
      if (compileError) {
        OCCA_FORCE_ERROR("Error compiling [" << kernelName << "],"
                         " Command: [" << sCommand << ']');
      }
      //================================
    }

    modeKernel_t* device::buildOKLKernelFromBinary(const hash_t kernelHash,
                                                   const std::string &hashDir,
                                                   const std::string &kernelName,
                                                   lang::sourceMetadata_t &launcherMetadata,
                                                   lang::sourceMetadata_t &deviceMetadata,
                                                   const occa::properties &kernelProps,
                                                   io::lock_t lock) {

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

      k.launcherKernel = buildLauncherKernel(kernelHash,
                                             hashDir,
                                             kernelName,
                                             launcherMetadata);

      // Find device kernels
      orderedKernelMetadata launchedKernelsMetadata = getLaunchedKernelsMetadata(
        kernelName,
        deviceMetadata
      );

      const int launchedKernelsCount = (int) launchedKernelsMetadata.size();
      for (int i = 0; i < launchedKernelsCount; ++i) {
        lang::kernelMetadata_t &metadata = launchedKernelsMetadata[i];

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
        cuKernel->metadata = metadata;
        k.deviceKernels.push_back(cuKernel);
      }

      return &k;
    }

    modeKernel_t* device::buildKernelFromBinary(const std::string &filename,
                                                const std::string &kernelName,
                                                const occa::properties &kernelProps) {
      CUmodule cuModule = NULL;
      CUfunction cuFunction = NULL;

      OCCA_CUDA_ERROR("Kernel [" + kernelName + "]: Loading Module",
                      cuModuleLoad(&cuModule, filename.c_str()));

      OCCA_CUDA_ERROR("Kernel [" + kernelName + "]: Loading Function",
                      cuModuleGetFunction(&cuFunction, cuModule, kernelName.c_str()));

      return new kernel(this,
                        kernelName,
                        filename,
                        cuModule,
                        cuFunction,
                        kernelProps);
    }
    //==================================

    //---[ Memory ]---------------------
    modeMemory_t* device::malloc(const udim_t bytes,
                                 const void *src,
                                 const occa::properties &props) {
      if (props.get("mapped", false)) {
        return mappedAlloc(bytes, src, props);
      }
      if (props.get("unified", false)) {
        return unifiedAlloc(bytes, src, props);
      }

      cuda::memory &mem = *(new cuda::memory(this, bytes, props));

      OCCA_CUDA_ERROR("Device: Setting Context",
                      cuCtxSetCurrent(cuContext));

      OCCA_CUDA_ERROR("Device: malloc",
                      cuMemAlloc(&(mem.cuPtr), bytes));

      if (src != NULL) {
        mem.copyFrom(src, bytes, 0);
      }
      return &mem;
    }

    modeMemory_t* device::mappedAlloc(const udim_t bytes,
                                      const void *src,
                                      const occa::properties &props) {

      cuda::memory &mem = *(new cuda::memory(this, bytes, props));

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

    modeMemory_t* device::unifiedAlloc(const udim_t bytes,
                                       const void *src,
                                       const occa::properties &props) {
      cuda::memory &mem = *(new cuda::memory(this, bytes, props));
#if CUDA_VERSION >= 8000
      mem.isUnified = true;

      const unsigned int flags = (props.get("attached_host", false) ?
                                  CU_MEM_ATTACH_HOST : CU_MEM_ATTACH_GLOBAL);

      OCCA_CUDA_ERROR("Device: Setting Context",
                      cuCtxSetCurrent(cuContext));
      OCCA_CUDA_ERROR("Device: Unified alloc",
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
