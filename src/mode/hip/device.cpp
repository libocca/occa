#include <occa/defines.hpp>

#if OCCA_HIP_ENABLED

#include <occa/core/base.hpp>
#include <occa/io/output.hpp>
#include <occa/tools/env.hpp>
#include <occa/tools/misc.hpp>
#include <occa/tools/sys.hpp>
#include <occa/mode/hip/device.hpp>
#include <occa/mode/hip/kernel.hpp>
#include <occa/mode/hip/memory.hpp>
#include <occa/mode/hip/utils.hpp>
#include <occa/mode/serial/device.hpp>
#include <occa/mode/serial/kernel.hpp>
#include <occa/lang/kernelMetadata.hpp>
#include <occa/lang/primitive.hpp>
#include <occa/lang/mode/hip.hpp>

namespace occa {
  namespace hip {
    device::device(const occa::properties &properties_) :
      occa::launchedModeDevice_t(properties_) {

      hipDeviceProp_t props;
      if (!properties.has("wrapped")) {
        OCCA_ERROR("[HIP] device not given a [device_id] integer",
                   properties.has("device_id") &&
                   properties["device_id"].isNumber());

        const int deviceID = properties.get<int>("device_id");

        OCCA_HIP_ERROR("Device: Creating Device",
                       hipDeviceGet(&hipDevice, deviceID));

        OCCA_HIP_ERROR("Device: Creating Context",
                       hipCtxCreate(&hipContext, 0, hipDevice));

        OCCA_HIP_ERROR("Getting device properties",
                       hipGetDeviceProperties(&props, deviceID));
      }

      p2pEnabled = false;

      std::string compiler = properties["kernel/compiler"];
      std::string compilerFlags = properties["kernel/compilerFlags"];

      if (!compiler.size()) {
        if (env::var("OCCA_HIP_COMPILER").size()) {
          compiler = env::var("OCCA_HIP_COMPILER");
        } else {
          compiler = "hipcc";
        }
      }

      if (!compilerFlags.size()) {
        compilerFlags = env::var("OCCA_HIP_COMPILER_FLAGS");
      }

      properties["kernel/compiler"]      = compiler;
      properties["kernel/compilerFlags"] = compilerFlags;

      OCCA_HIP_ERROR("Device: Getting HIP Device Arch",
                     hipDeviceComputeCapability(&archMajorVersion,
                                                &archMinorVersion,
                                                hipDevice) );

      archMajorVersion = properties.get("hip/arch/major", archMajorVersion);
      archMinorVersion = properties.get("hip/arch/minor", archMinorVersion);

      properties["kernel/target"] = toString(props.gcnArch);
    }

    device::~device() {
      if (hipContext) {
        OCCA_HIP_ERROR("Device: Freeing Context",
                       hipCtxDestroy(hipContext) );
        hipContext = NULL;
      }
    }

    void device::finish() const {
      OCCA_HIP_ERROR("Device: Finish",
                     hipStreamSynchronize(*((hipStream_t*) currentStream)) );
      hipDeviceSynchronize();
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
    stream_t device::createStream(const occa::properties &props) {
      hipStream_t *retStream = new hipStream_t;

      OCCA_HIP_ERROR("Device: Setting Context",
                     hipCtxSetCurrent(hipContext));
      OCCA_HIP_ERROR("Device: createStream",
                     hipStreamCreate(retStream));

      return retStream;
    }

    void device::freeStream(stream_t s) const {
      OCCA_HIP_ERROR("Device: freeStream",
                     hipStreamDestroy( *((hipStream_t*) s) ));
      delete (hipStream_t*) s;
    }

    streamTag device::tagStream() const {
      streamTag ret;

      OCCA_HIP_ERROR("Device: Setting Context",
                     hipCtxSetCurrent(hipContext));
      OCCA_HIP_ERROR("Device: Tagging Stream (Creating Tag)",
                     hipEventCreate(&hip::event(ret)));
      OCCA_HIP_ERROR("Device: Tagging Stream",
                     hipEventRecord(hip::event(ret), 0));

      return ret;
    }

    void device::waitFor(streamTag tag) const {
      OCCA_HIP_ERROR("Device: Waiting For Tag",
                     hipEventSynchronize(hip::event(tag)));
    }

    double device::timeBetween(const streamTag &startTag,
                               const streamTag &endTag) const {
      OCCA_HIP_ERROR("Device: Waiting for endTag",
                     hipEventSynchronize(hip::event(endTag)));

      float msTimeTaken;
      OCCA_HIP_ERROR("Device: Timing Between Tags",
                     hipEventElapsedTime(&msTimeTaken, hip::event(startTag), hip::event(endTag)));

      return (double) (1.0e-3 * (double) msTimeTaken);
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
      lang::kernelMetadataMap &launcherMetadata,
      lang::kernelMetadataMap &deviceMetadata,
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
      if (kernelProps.get<std::string>("compiler_flags").find("-t gfx") == std::string::npos) {
        std::stringstream ss;
        std::string arch = kernelProps["target"];
        if (arch.size()) {
          ss << " -t gfx" << arch << ' ';
          kernelProps["compiler_flags"] += ss.str();
        }
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

      std::stringstream command;

      //---[ Compiling Command ]--------
      command.str("");
      command << kernelProps["compiler"]
              << " --genco "
              << " "       << sourceFilename
              << " -o "    << binaryFilename
              << ' ' << kernelProps["compiler_flags"]
#if (OCCA_OS == OCCA_WINDOWS_OS)
              << " -D OCCA_OS=OCCA_WINDOWS_OS -D _MSC_VER=1800"
#endif
        ;

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
                                                   lang::kernelMetadataMap &launcherMetadata,
                                                   lang::kernelMetadataMap &deviceMetadata,
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

      k.launcherKernel = buildLauncherKernel(hashDir,
                                             kernelName,
                                             launcherMetadata[kernelName]);

      // Find clKernels
      typedef std::map<int, lang::kernelMetadata> kernelOrderMap;
      kernelOrderMap hipKernelMetadata;

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
        hipKernelMetadata[number] = metadata;
      }

      kernelOrderMap::iterator oit = hipKernelMetadata.begin();
      while (oit != hipKernelMetadata.end()) {
        lang::kernelMetadata &metadata = oit->second;

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
        hipKernel->dontUseRefs();
        k.deviceKernels.push_back(hipKernel);

        ++oit;
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

      OCCA_HIP_ERROR("Device: Setting Context",
                     hipCtxSetCurrent(hipContext));

      OCCA_HIP_ERROR("Device: malloc",
                     hipMalloc(&(mem.hipPtr), bytes));

      if (src != NULL) {
        mem.copyFrom(src, bytes, 0);
      }
      return &mem;
    }

    modeMemory_t* device::mappedAlloc(const udim_t bytes,
                                      const void *src,
                                      const occa::properties &props) {

      hip::memory &mem = *(new hip::memory(this, bytes, props));

      OCCA_HIP_ERROR("Device: Setting Context",
                     hipCtxSetCurrent(hipContext));
      OCCA_HIP_ERROR("Device: malloc host",
                     hipHostMalloc((void**) &(mem.mappedPtr), bytes));
      OCCA_HIP_ERROR("Device: get device pointer from host",
                     hipHostGetDevicePointer(&(mem.hipPtr),
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

#endif
