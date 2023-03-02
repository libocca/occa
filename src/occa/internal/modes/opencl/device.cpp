#include <occa/core/base.hpp>
#include <occa/internal/utils/env.hpp>
#include <occa/internal/utils/sys.hpp>
#include <occa/internal/modes/opencl/device.hpp>
#include <occa/internal/modes/opencl/kernel.hpp>
#include <occa/internal/modes/opencl/memory.hpp>
#include <occa/internal/modes/opencl/memoryPool.hpp>
#include <occa/internal/modes/opencl/buffer.hpp>
#include <occa/internal/modes/opencl/stream.hpp>
#include <occa/internal/modes/opencl/streamTag.hpp>
#include <occa/internal/modes/opencl/utils.hpp>
#include <occa/internal/modes/serial/device.hpp>
#include <occa/internal/modes/serial/kernel.hpp>
#include <occa/internal/lang/kernelMetadata.hpp>
#include <occa/types/primitive.hpp>
#include <occa/internal/lang/modes/opencl.hpp>

namespace occa {
  namespace opencl {
    device::device(const occa::json &properties_) :
      occa::launchedModeDevice_t(properties_) {

      if (!properties.has("wrapped")) {
        cl_int error;
        OCCA_ERROR("[OpenCL] device not given a [platform_id] integer",
                   properties.has("platform_id") &&
                   properties["platform_id"].isNumber());

        OCCA_ERROR("[OpenCL] device not given a [device_id] integer",
                   properties.has("device_id") &&
                   properties["device_id"].isNumber());

        platformID = properties.get<int>("platform_id");
        deviceID   = properties.get<int>("device_id");

        clDevice = opencl::deviceID(platformID, deviceID);

        clContext = clCreateContext(NULL, 1, &clDevice, NULL, NULL, &error);
        OCCA_OPENCL_ERROR("Device: Creating Context", error);
      }

      occa::json &kernelProps = properties["kernel"];
      std::string compilerFlags;

      // Use "-cl-opt-disable" for debug-mode
      if (kernelProps.has("compiler_flags")) {
        compilerFlags = (std::string) kernelProps["compiler_flags"];
      } else if (env::var("OCCA_OPENCL_COMPILER_FLAGS").size()) {
        compilerFlags = env::var("OCCA_OPENCL_COMPILER_FLAGS");
      }

      std::string ocl_c_ver = "2.0";
      if (env::var("OCCA_OPENCL_C_VERSION").size()) {
        ocl_c_ver = env::var("OCCA_OPENCL_C_VERSION");
      }
      compilerFlags += " -cl-std=CL" + ocl_c_ver;

      kernelProps["compiler_flags"] = compilerFlags;
    }

    device::~device() {
      if (clContext) {
        OCCA_OPENCL_ERROR("Device: Freeing Context",
                          clReleaseContext(clContext) );
        clContext = NULL;
      }
    }

    bool device::hasSeparateMemorySpace() const {
      return true;
    }

    hash_t device::hash() const {
      if (!hash_.initialized) {
        std::stringstream ss;
        ss << "platform name: " << opencl::platformName(platformID)
          << " platform vendor: " << opencl::platformVendor(platformID)
          << " platform version: " << opencl::platformVersion(platformID)
          << " device name: " << opencl::deviceName(platformID,deviceID)
          << " device vendor: " << opencl::deviceVendor(platformID,deviceID)
          << " device version: " << opencl::deviceVersion(platformID,deviceID);
        hash_ = occa::hash(ss.str());
      }
      return hash_;
    }

    hash_t device::kernelHash(const occa::json &props) const {
      return occa::hash(props["compiler_flags"]);
    }

    lang::okl::withLauncher* device::createParser(const occa::json &props) const {
      return new lang::okl::openclParser(props);
    }

    //---[ Stream ]---------------------
    modeStream_t* device::createStream(const occa::json &props) {
      cl_int error;
#ifdef CL_VERSION_2_0
      cl_queue_properties clProps[] = {CL_QUEUE_PROPERTIES,
                                       CL_QUEUE_PROFILING_ENABLE, 0};
      cl_command_queue commandQueue = clCreateCommandQueueWithProperties(clContext,
                                                           clDevice,
                                                           clProps,
                                                           &error);
#else
      cl_command_queue commandQueue = clCreateCommandQueue(clContext,
                                                           clDevice,
                                                           CL_QUEUE_PROFILING_ENABLE,
                                                           &error);
#endif
      OCCA_OPENCL_ERROR("Device: createStream", error);

      return new stream(this, props, commandQueue);
    }

    modeStream_t* device::wrapStream(void* ptr, const occa::json &props) {
      OCCA_ERROR("A nullptr was passed to opencl::device::wrapStream",nullptr != ptr);

      cl_command_queue commandQueue = *static_cast<cl_command_queue*>(ptr);
      OCCA_OPENCL_ERROR("Device: Retaining Command Queue",
                        clRetainCommandQueue(commandQueue));

      return new stream(this, props, commandQueue);
    }

    occa::streamTag device::tagStream() {
      cl_event clEvent = NULL;

#ifdef CL_VERSION_1_2
      OCCA_OPENCL_ERROR("Device: Tagging Stream",
                        clEnqueueBarrierWithWaitList(getCommandQueue(),
                                                    0, NULL, &clEvent));
#else
      OCCA_OPENCL_ERROR("Device: Tagging Stream",
                        clEnqueueBarrier(getCommandQueue(),
                                        &clEvent));
#endif

      return new occa::opencl::streamTag(this, clEvent);
    }

    void device::waitFor(occa::streamTag tag) {
      occa::opencl::streamTag *clTag = (
        dynamic_cast<occa::opencl::streamTag*>(tag.getModeStreamTag())
      );
      OCCA_OPENCL_ERROR("Device: Waiting For Tag",
                        clWaitForEvents(1, &(clTag->clEvent)));
    }

    double device::timeBetween(const occa::streamTag &startTag,
                               const occa::streamTag &endTag) {
      occa::opencl::streamTag *clStartTag = (
        dynamic_cast<occa::opencl::streamTag*>(startTag.getModeStreamTag())
      );
      occa::opencl::streamTag *clEndTag = (
        dynamic_cast<occa::opencl::streamTag*>(endTag.getModeStreamTag())
      );

      waitFor(endTag);

      return (clEndTag->endTime() - clStartTag->startTime());
    }

    cl_command_queue& device::getCommandQueue() const {
      occa::opencl::stream *stream = (occa::opencl::stream*) currentStream.getModeStream();
      return stream->commandQueue;
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
      info_t clInfo;
      clInfo.clDevice  = clDevice;
      clInfo.clContext = clContext;

      // Build OpenCL program
      std::string source = io::read(sourceFilename, enums::FILE_TYPE_BINARY);

      opencl::buildProgramFromSource(clInfo,
                                     source,
                                     kernelName,
                                     kernelProps["compiler_flags"],
                                     sourceFilename,
                                     kernelProps);

      opencl::saveProgramBinary(clInfo,
                                binaryFilename);

      if (usingOkl) {
        return buildOKLKernelFromBinary(clInfo,
                                        kernelHash,
                                        hashDir,
                                        kernelName,
                                        sourceFilename,
                                        binaryFilename,
                                        launcherMetadata,
                                        deviceMetadata,
                                        kernelProps);
      }

      // Regular OpenCL Kernel
      opencl::buildKernelFromProgram(clInfo,
                                     kernelName);
      return new kernel(this,
                        kernelName,
                        sourceFilename,
                        clDevice,
                        clInfo.clKernel,
                        kernelProps);
    }

    modeKernel_t* device::buildOKLKernelFromBinary(
      const hash_t kernelHash,
      const std::string &hashDir,
      const std::string &kernelName,
      const std::string &sourceFilename,
      const std::string &binaryFilename,
      lang::sourceMetadata_t &launcherMetadata,
      lang::sourceMetadata_t &deviceMetadata,
      const occa::json &kernelProps
    ) {
      info_t clInfo;
      clInfo.clDevice  = clDevice;
      clInfo.clContext = clContext;

      return buildOKLKernelFromBinary(clInfo,
                                      kernelHash,
                                      hashDir,
                                      kernelName,
                                      sourceFilename,
                                      binaryFilename,
                                      launcherMetadata,
                                      deviceMetadata,
                                      kernelProps);
    }

    modeKernel_t* device::buildOKLKernelFromBinary(
      info_t &clInfo,
      const hash_t kernelHash,
      const std::string &hashDir,
      const std::string &kernelName,
      const std::string &sourceFilename,
      const std::string &binaryFilename,
      lang::sourceMetadata_t &launcherMetadata,
      lang::sourceMetadata_t &deviceMetadata,
      const occa::json &kernelProps
    ) {
      if (!clInfo.clProgram) {
        opencl::buildProgramFromBinary(clInfo,
                                       binaryFilename,
                                       kernelName,
                                       properties["compiler_flags"]);
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
      if (!k.launcherKernel) {
        delete &k;
        return NULL;
      }

      // Find device kernels
      orderedKernelMetadata launchedKernelsMetadata = getLaunchedKernelsMetadata(
        kernelName,
        deviceMetadata
      );

      const int launchedKernelsCount = (int) launchedKernelsMetadata.size();
      for (int i = 0; i < launchedKernelsCount; ++i) {
        lang::kernelMetadata_t &metadata = launchedKernelsMetadata[i];
        opencl::buildKernelFromProgram(clInfo,
                                       metadata.name);

        kernel *clKernel = new kernel(this,
                                      metadata.name,
                                      sourceFilename,
                                      clDevice,
                                      clInfo.clKernel,
                                      kernelProps);
        clKernel->metadata = metadata;
        k.deviceKernels.push_back(clKernel);
      }

      return &k;
    }

    modeKernel_t* device::buildKernelFromBinary(const std::string &filename,
                                                const std::string &kernelName,
                                                const occa::json &kernelProps) {

      info_t clInfo;
      clInfo.clDevice  = clDevice;
      clInfo.clContext = clContext;

      opencl::buildProgramFromBinary(clInfo,
                                     filename,
                                     kernelName,
                                     properties["compiler_flags"]);

      opencl::buildKernelFromProgram(clInfo,
                                     kernelName);

      return new kernel(this,
                        kernelName,
                        filename,
                        clDevice,
                        clInfo.clKernel,
                        kernelProps);
    }
    //==================================

    //---[ Memory ]---------------------
    modeMemory_t* device::malloc(const udim_t bytes,
                                 const void *src,
                                 const occa::json &props) {

      //create allocation
      buffer *buf = new opencl::buffer(this, bytes, props);

      buf->malloc(bytes);

      //create slice
      memory *mem = new opencl::memory(buf, bytes, 0);

      if (src) {
        mem->copyFrom(src, bytes, 0, props);
      }

      return mem;
    }

    modeMemory_t* device::wrapMemory(const void *ptr,
                                     const udim_t bytes,
                                     const occa::json &props) {
      //create allocation
      buffer *buf = new opencl::buffer(this, bytes, props);
      buf->wrapMemory(ptr, bytes);

      return new opencl::memory(buf, bytes, 0);
    }

    modeMemoryPool_t* device::createMemoryPool(const occa::json &props) {
      return new opencl::memoryPool(this, props);
    }

    udim_t device::memorySize() const {
      return opencl::deviceGlobalMemSize(clDevice);
    }
    //==================================

    void* device::unwrap() {
      return static_cast<void*>(&clDevice);
    }
  }
}
