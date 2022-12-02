#include <occa/core/base.hpp>
#include <occa/internal/io/output.hpp>
#include <occa/internal/utils/env.hpp>
#include <occa/internal/utils/misc.hpp>
#include <occa/internal/utils/sys.hpp>
#include <occa/internal/modes/hip/device.hpp>
#include <occa/internal/modes/hip/kernel.hpp>
#include <occa/internal/modes/hip/buffer.hpp>
#include <occa/internal/modes/hip/memory.hpp>
#include <occa/internal/modes/hip/memoryPool.hpp>
#include <occa/internal/modes/hip/stream.hpp>
#include <occa/internal/modes/hip/streamTag.hpp>
#include <occa/internal/modes/hip/utils.hpp>
#include <occa/internal/modes/serial/device.hpp>
#include <occa/internal/modes/serial/kernel.hpp>
#include <occa/internal/lang/kernelMetadata.hpp>
#include <occa/types/primitive.hpp>
#include <occa/internal/lang/modes/hip.hpp>

namespace occa {
  namespace hip {
    device::device(const occa::json &properties_) :
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

      if (kernelProps.get<std::string>("compiler_flags").size()) {
        compilerFlags = (std::string) kernelProps["compiler_flags"];
      } else if (env::var("OCCA_HIP_COMPILER_FLAGS").size()) {
        compilerFlags = env::var("OCCA_HIP_COMPILER_FLAGS");
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
#if HIP_VERSION >= 305
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

    hash_t device::kernelHash(const occa::json &props) const {
      return (
        occa::hash(props["compiler"])
        ^ props["compiler_flags"]
        ^ props["compiler_env_script"]
      );
    }

    lang::okl::withLauncher* device::createParser(const occa::json &props) const {
      return new lang::okl::hipParser(props);
    }

    //---[ Stream ]---------------------
    modeStream_t* device::createStream(const occa::json &props) {
      hipStream_t hipStream = NULL;

      OCCA_HIP_ERROR("Device: Setting Device",
                     hipSetDevice(deviceID));
      if (props.get<bool>("nonblocking", false)) {
        OCCA_HIP_ERROR("Device: createStream - NonBlocking",
                       hipStreamCreateWithFlags(&hipStream, hipStreamNonBlocking));
      } else {
        OCCA_HIP_ERROR("Device: createStream",
                       hipStreamCreate(&hipStream));
      }

      return new stream(this, props, hipStream);
    }

    modeStream_t* device::wrapStream(void* ptr, const occa::json &props) {
      OCCA_ERROR("A nullptr was passed to hip::device::wrapStream",nullptr != ptr);
      hipStream_t hipStream = *static_cast<hipStream_t*>(ptr);
      return new stream(this, props, hipStream);
    }

    occa::streamTag device::tagStream() {
      hipEvent_t hipEvent = NULL;

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

      float msTimeTaken = 0.0;
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
      const occa::json &kernelProps
    ) {
      compileKernel(hashDir,
                    kernelName,
                    sourceFilename,
                    binaryFilename,
                    kernelProps);

      if (usingOkl) {
        return buildOKLKernelFromBinary(kernelHash,
                                        hashDir,
                                        kernelName,
                                        sourceFilename,
                                        binaryFilename,
                                        launcherMetadata,
                                        deviceMetadata,
                                        kernelProps);
      }

      // Regular HIP Kernel
      hipModule_t hipModule = NULL;
      hipFunction_t hipFunction = NULL;

      OCCA_HIP_ERROR("Kernel [" + kernelName + "]: Loading Module",
                     hipModuleLoad(&hipModule, binaryFilename.c_str()));

      OCCA_HIP_ERROR("Kernel [" + kernelName + "]: Loading Function",
                     hipModuleGetFunction(&hipFunction,
                                          hipModule,
                                          kernelName.c_str()));

      return new kernel(this,
                        kernelName,
                        sourceFilename,
                        hipModule,
                        hipFunction,
                        kernelProps);
    }

    void device::setArchCompilerFlags(occa::json &kernelProps) {
      const std::string hipccCompilerFlags = (
        kernelProps.get<std::string>("hipcc_compiler_flags")
      );

      if (hipccCompilerFlags.find("-arch=sm") == std::string::npos &&
#if HIP_VERSION >= 305
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
                               const std::string &sourceFilename,
                               const std::string &binaryFilename,
                               const occa::json &kernelProps) {

      occa::json allProps = kernelProps;
      const bool verbose = allProps.get("verbose", false);

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
#if defined(__HIP_PLATFORM_NVCC___) || (HIP_VERSION >= 305)
              << ' ' << compilerFlags
#else
              << " -f=\\\"" << compilerFlags << "\\\""
#endif
              << ' ' << hipccCompilerFlags
#if defined(__HIP_PLATFORM_NVCC___) || (HIP_VERSION >= 305)
              << " -I"        << env::OCCA_DIR << "include"
              << " -I"        << env::OCCA_INSTALL_DIR << "include"
#endif
              /* NC: hipcc doesn't seem to like linking a library in */
              //<< " -L"        << env::OCCA_INSTALL_DIR << "lib -locca"
              << ' '    << sourceFilename
              << " -o " << binaryFilename
              << " 2>&1";

      const std::string &sCommand = command.str();
      if (verbose) {
        io::stdout << "Compiling [" << kernelName << "]\n" << sCommand << "\n";
      }

      std::string commandOutput;
      const int commandExitCode = sys::call(
        sCommand.c_str(),
        commandOutput
      );

      if (commandExitCode) {
        OCCA_FORCE_ERROR(
          "Error compiling [" << kernelName << "],"
          " Command: [" << sCommand << "] exited with code " << commandExitCode << "\n\n"
          << "Output:\n\n"
          << commandOutput << "\n"
        );
      } else if (verbose) {
          io::stdout << "Output:\n\n" << commandOutput << "\n";
      }
      
      io::sync(binaryFilename);
    }

    modeKernel_t* device::buildOKLKernelFromBinary(const hash_t kernelHash,
                                                   const std::string &hashDir,
                                                   const std::string &kernelName,
                                                   const std::string &sourceFilename,
                                                   const std::string &binaryFilename,
                                                   lang::sourceMetadata_t &launcherMetadata,
                                                   lang::sourceMetadata_t &deviceMetadata,
                                                   const occa::json &kernelProps) {
      hipModule_t hipModule = NULL;

      OCCA_HIP_ERROR("Kernel [" + kernelName + "]: Loading Module",
                     hipModuleLoad(&hipModule, binaryFilename.c_str()));

      // Create wrapper kernel and set launcherKernel
      kernel &k = *(new kernel(this,
                               kernelName,
                               sourceFilename,
                               hipModule,
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

        hipFunction_t hipFunction = NULL;

        OCCA_HIP_ERROR("Kernel [" + metadata.name + "]: Loading Function",
                       hipModuleGetFunction(&hipFunction,
                                            hipModule,
                                            metadata.name.c_str()));

        kernel *hipKernel = new kernel(this,
                                       metadata.name,
                                       sourceFilename,
                                       hipFunction,
                                       kernelProps);
        hipKernel->metadata = metadata;
        k.deviceKernels.push_back(hipKernel);
      }

      return &k;
    }

    modeKernel_t* device::buildKernelFromBinary(const std::string &filename,
                                                const std::string &kernelName,
                                                const occa::json &kernelProps) {

      hipModule_t hipModule = NULL;
      hipFunction_t hipFunction = NULL;

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
                                 const occa::json &props) {

      OCCA_HIP_ERROR("Device: Setting Device",
                     hipSetDevice(deviceID));

      buffer *buf = new hip::buffer(this, bytes, props);

      //create allocation
      buf->malloc(bytes);

      //create slice
      memory *mem = new hip::memory(buf, bytes, 0);

      if (src != NULL)
        mem->copyFrom(src, bytes, 0, props);

      return mem;
    }

    modeMemory_t* device::wrapMemory(const void *ptr,
                                     const udim_t bytes,
                                     const occa::json &props) {
      //create allocation
      buffer *buf = new hip::buffer(this, bytes, props);

      buf->wrapMemory(ptr, bytes);

      return new hip::memory(buf, bytes, 0);
    }

    modeMemoryPool_t* device::createMemoryPool(const occa::json &props) {
      return new hip::memoryPool(this, props);
    }

    udim_t device::memorySize() const {
      return hip::getDeviceMemorySize(hipDevice);
    }
    //==================================

    void* device::unwrap() {
      return static_cast<void*>(&hipDevice);
    }
  }
}
