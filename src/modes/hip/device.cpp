#include <occa/core/base.hpp>
#include <occa/io/output.hpp>
#include <occa/tools/env.hpp>
#include <occa/tools/misc.hpp>
#include <occa/tools/sys.hpp>
#include <occa/modes/hip/device.hpp>
#include <occa/modes/hip/kernel.hpp>
#include <occa/modes/hip/memory.hpp>
#include <occa/modes/hip/stream.hpp>
#include <occa/modes/hip/streamTag.hpp>
#include <occa/modes/hip/utils.hpp>
#include <occa/modes/serial/device.hpp>
#include <occa/modes/serial/kernel.hpp>
#include <occa/lang/kernelMetadata.hpp>
#include <occa/lang/primitive.hpp>
#include <occa/lang/modes/hip.hpp>

namespace occa {
  namespace hip {
    device::device(const occa::properties &properties_) :
      occa::launchedModeDevice_t(properties_) {

      hipDeviceProp_t hipProps;
      if (!properties.has("wrapped")) {
        OCCA_ERROR("[HIP] device not given a [device_id] integer",
                   properties.has("device_id") &&
                   properties["device_id"].isNumber());

        deviceID = properties.get<int>("device_id");

        OCCA_HIP_ERROR("Device: Creating Device",
                       hipDeviceGet(&hipDevice, deviceID));

        OCCA_HIP_ERROR("Device: Setting Device",
                       hipSetDevice(deviceID));

        OCCA_HIP_ERROR("Getting device properties",
                       hipGetDeviceProperties(&hipProps, deviceID));
      }

      p2pEnabled = false;

      occa::json &kernelProps = properties["kernel"];
      std::string compiler, compilerFlags;

      if (env::var("OCCA_HIP_COMPILER").size()) {
        compiler = env::var("OCCA_HIP_COMPILER");
      } else if (kernelProps.get<std::string>("compiler").size()) {
        compiler = (std::string) kernelProps["compiler"];
      } else {
        compiler = "hipcc";
      }

      if (env::var("OCCA_HIP_COMPILER_FLAGS").size()) {
        compilerFlags = env::var("OCCA_HIP_COMPILER_FLAGS");
      } else if (kernelProps.get<std::string>("compiler_flags").size()) {
        compilerFlags = (std::string) kernelProps["compiler_flags"];
      } else {
        compilerFlags = "-O3";
      }

      kernelProps["compiler"]       = compiler;
      kernelProps["compiler_flags"] = compilerFlags;

      archMajorVersion = kernelProps.get<int>("arch/major", hipProps.major);
      archMinorVersion = kernelProps.get<int>("arch/minor", hipProps.minor);

      std::string arch = getDeviceArch(deviceID, archMajorVersion, archMinorVersion);
      std::string archFlag;
      if (startsWith(arch, "sm_")) {
        archFlag = " -arch=" + arch;
      } else if (startsWith(arch, "gfx")) {
#ifdef __HIP_ROCclr__
        archFlag = " --amdgpu-target=" + arch;
#else
        archFlag = " -t " + arch;
#endif
      } else {
        OCCA_FORCE_ERROR("Unknown HIP arch");
      }

      kernelProps["compiler_flag_arch"] = archFlag;
    }

    device::~device() { }

    void device::finish() const {
      OCCA_HIP_ERROR("Device: Finish",
                     hipStreamSynchronize(getHipStream()));
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
      return new lang::okl::hipParser(props);
    }

    //---[ Stream ]---------------------
    modeStream_t* device::createStream(const occa::properties &props) {
      hipStream_t hipStream;

      OCCA_HIP_ERROR("Device: Setting Device",
                     hipSetDevice(deviceID));
      OCCA_HIP_ERROR("Device: createStream",
                     hipStreamCreate(&hipStream));

      return new stream(this, props, hipStream);
    }

    occa::streamTag device::tagStream() {
      hipEvent_t hipEvent;

      OCCA_HIP_ERROR("Device: Setting Device",
                     hipSetDevice(deviceID));
      OCCA_HIP_ERROR("Device: Tagging Stream (Creating Tag)",
                     hipEventCreate(&hipEvent));
      OCCA_HIP_ERROR("Device: Tagging Stream",
                     hipEventRecord(hipEvent, getHipStream()));

      return new occa::hip::streamTag(this, hipEvent);
    }

    void device::waitFor(occa::streamTag tag) {
      occa::hip::streamTag *hipTag = (
        dynamic_cast<occa::hip::streamTag*>(tag.getModeStreamTag())
      );
      OCCA_HIP_ERROR("Device: Waiting For Tag",
                     hipEventSynchronize(hipTag->hipEvent));
    }

    double device::timeBetween(const occa::streamTag &startTag,
                               const occa::streamTag &endTag) {
      occa::hip::streamTag *hipStartTag = (
        dynamic_cast<occa::hip::streamTag*>(startTag.getModeStreamTag())
      );
      occa::hip::streamTag *hipEndTag = (
        dynamic_cast<occa::hip::streamTag*>(endTag.getModeStreamTag())
      );

      waitFor(endTag);

      float msTimeTaken;
      OCCA_HIP_ERROR("Device: Timing Between Tags",
                     hipEventElapsedTime(&msTimeTaken,
                                         hipStartTag->hipEvent,
                                         hipEndTag->hipEvent));

      return (double) (1.0e-3 * (double) msTimeTaken);
    }

    hipStream_t& device::getHipStream() const {
      occa::hip::stream *stream = (occa::hip::stream*) currentStream.getModeStream();
      return stream->hipStream;
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

      // Regular HIP Kernel
      hipModule_t hipModule;
      hipFunction_t hipFunction;
      hipError_t error;

      error = hipModuleLoad(&hipModule, binaryFilename.c_str());
      if (error) {
        lock.release();
        OCCA_HIP_ERROR("Kernel [" + kernelName + "]: Loading Module",
                       error);
      }
      error = hipModuleGetFunction(&hipFunction,
                                   hipModule,
                                   kernelName.c_str());
      if (error) {
        lock.release();
        OCCA_HIP_ERROR("Kernel [" + kernelName + "]: Loading Function",
                       error);
      }

      return new kernel(this,
                        kernelName,
                        sourceFilename,
                        hipModule,
                        hipFunction,
                        kernelProps);
    }

    void device::setArchCompilerFlags(occa::properties &kernelProps) {
      const std::string hipccCompilerFlags = (
        kernelProps.get<std::string>("hipcc_compiler_flags")
      );

      if (hipccCompilerFlags.find("-arch=sm") == std::string::npos &&
#ifdef __HIP_ROCclr__
          hipccCompilerFlags.find("-t gfx") == std::string::npos
#else
          hipccCompilerFlags.find("--amdgpu-target=gfx") == std::string::npos
#endif
          ) {
        kernelProps["hipcc_compiler_flags"] += " ";
        kernelProps["hipcc_compiler_flags"] += kernelProps["compiler_flag_arch"];
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

      setArchCompilerFlags(allProps);

      const std::string compiler = allProps["compiler"];
      std::string compilerFlags = allProps["compiler_flags"];
      const std::string hipccCompilerFlags = allProps["hipcc_compiler_flags"];
      const bool compilingOkl = allProps.get("okl/enabled", true);

      if (!compilingOkl) {
        sys::addCompilerIncludeFlags(compilerFlags);
        sys::addCompilerLibraryFlags(compilerFlags);
      }

      std::stringstream command;
      if (allProps.has("compiler_env_script")) {
        command << allProps["compiler_env_script"] << " && ";
      }

      //---[ Compiling Command ]--------
      command << compiler
              << " --genco"
#if defined(__HIP_PLATFORM_NVCC___) || defined(__HIP_ROCclr__)
              << ' ' << compilerFlags
#else
              << " -f=\\\"" << compilerFlags << "\\\""
#endif
              << ' ' << hipccCompilerFlags
#if defined(__HIP_PLATFORM_NVCC___) || defined(__HIP_ROCclr__)
              << " -I"        << env::OCCA_DIR << "include"
              << " -I"        << env::OCCA_INSTALL_DIR << "include"
#endif
              /* NC: hipcc doesn't seem to like linking a library in */
              //<< " -L"        << env::OCCA_INSTALL_DIR << "lib -locca"
              << ' '    << sourceFilename
              << " -o " << binaryFilename;

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

      hipModule_t hipModule;
      hipError_t error;

      error = hipModuleLoad(&hipModule, binaryFilename.c_str());
      if (error) {
        lock.release();
        OCCA_HIP_ERROR("Kernel [" + kernelName + "]: Loading Module",
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

        hipFunction_t hipFunction;
        error = hipModuleGetFunction(&hipFunction,
                                     hipModule,
                                     metadata.name.c_str());
        if (error) {
          lock.release();
          OCCA_HIP_ERROR("Kernel [" + metadata.name + "]: Loading Function",
                         error);
        }

        kernel *hipKernel = new kernel(this,
                                       metadata.name,
                                       sourceFilename,
                                       hipModule,
                                       hipFunction,
                                       kernelProps);
        hipKernel->metadata = metadata;
        k.deviceKernels.push_back(hipKernel);
      }

      return &k;
    }

    modeKernel_t* device::buildKernelFromBinary(const std::string &filename,
                                                const std::string &kernelName,
                                                const occa::properties &kernelProps) {

      hipModule_t hipModule;
      hipFunction_t hipFunction;

      OCCA_HIP_ERROR("Kernel [" + kernelName + "]: Loading Module",
                     hipModuleLoad(&hipModule, filename.c_str()));

      OCCA_HIP_ERROR("Kernel [" + kernelName + "]: Loading Function",
                     hipModuleGetFunction(&hipFunction, hipModule, kernelName.c_str()));

      return new kernel(this,
                        kernelName,
                        filename,
                        hipModule,
                        hipFunction,
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

      hip::memory &mem = *(new hip::memory(this, bytes, props));

      OCCA_HIP_ERROR("Device: Setting Device",
                     hipSetDevice(deviceID));

      OCCA_HIP_ERROR("Device: malloc",
                     hipMalloc((void**) &(mem.hipPtr), bytes));

      if (src != NULL) {
        mem.copyFrom(src, bytes, 0);
      }
      return &mem;
    }

    modeMemory_t* device::mappedAlloc(const udim_t bytes,
                                      const void *src,
                                      const occa::properties &props) {

      hip::memory &mem = *(new hip::memory(this, bytes, props));

      OCCA_HIP_ERROR("Device: Setting Device",
                     hipSetDevice(deviceID));
      OCCA_HIP_ERROR("Device: malloc host",
                     hipHostMalloc((void**) &(mem.mappedPtr), bytes));
      OCCA_HIP_ERROR("Device: get device pointer from host",
                     hipHostGetDevicePointer((void**) &(mem.hipPtr),
                                             mem.mappedPtr,
                                             0));

      if (src != NULL) {
        ::memcpy(mem.mappedPtr, src, bytes);
      }
      return &mem;
    }

    udim_t device::memorySize() const {
      return hip::getDeviceMemorySize(hipDevice);
    }
    //==================================
  }
}
