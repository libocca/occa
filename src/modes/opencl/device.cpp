#include <occa/core/base.hpp>
#include <occa/tools/env.hpp>
#include <occa/tools/sys.hpp>
#include <occa/modes/opencl/device.hpp>
#include <occa/modes/opencl/kernel.hpp>
#include <occa/modes/opencl/memory.hpp>
#include <occa/modes/opencl/stream.hpp>
#include <occa/modes/opencl/streamTag.hpp>
#include <occa/modes/opencl/utils.hpp>
#include <occa/modes/serial/device.hpp>
#include <occa/modes/serial/kernel.hpp>
#include <occa/lang/kernelMetadata.hpp>
#include <occa/lang/primitive.hpp>
#include <occa/lang/modes/opencl.hpp>

namespace occa {
  namespace opencl {
    device::device(const occa::properties &properties_) :
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

      kernelProps["compiler_flags"] = compilerFlags;
    }

    device::~device() {
      if (clContext) {
        OCCA_OPENCL_ERROR("Device: Freeing Context",
                          clReleaseContext(clContext) );
        clContext = NULL;
      }
    }

    void device::finish() const {
      OCCA_OPENCL_ERROR("Device: Finish",
                        clFinish(getCommandQueue()));
    }

    bool device::hasSeparateMemorySpace() const {
      return true;
    }

    hash_t device::hash() const {
      if (!hash_.initialized) {
        std::stringstream ss;
        ss << "platform: " << platformID << ' '
           << "device: " << deviceID;
        hash_ = occa::hash(ss.str());
      }
      return hash_;
    }

    hash_t device::kernelHash(const occa::properties &props) const {
      return occa::hash(props["compiler_flags"]);
    }

    lang::okl::withLauncher* device::createParser(const occa::properties &props) const {
      return new lang::okl::openclParser(props);
    }

    //---[ Stream ]---------------------
    modeStream_t* device::createStream(const occa::properties &props) {
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

    occa::streamTag device::tagStream() {
      cl_event clEvent;

#ifdef CL_VERSION_1_2
      OCCA_OPENCL_ERROR("Device: Tagging Stream",
                        clEnqueueMarkerWithWaitList(getCommandQueue(),
                                                    0, NULL, &clEvent));
#else
      OCCA_OPENCL_ERROR("Device: Tagging Stream",
                        clEnqueueMarker(getCommandQueue(),
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

      finish();

      return (clEndTag->getTime() - clStartTag->getTime());
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
      const occa::properties &kernelProps,
      io::lock_t lock
    ) {
      info_t clInfo;
      clInfo.clDevice  = clDevice;
      clInfo.clContext = clContext;

      // Build OpenCL program
      std::string source = io::read(sourceFilename, true);

      opencl::buildProgramFromSource(clInfo,
                                     source,
                                     kernelName,
                                     kernelProps["compiler_flags"],
                                     sourceFilename,
                                     kernelProps,
                                     lock);

      opencl::saveProgramBinary(clInfo,
                                binaryFilename,
                                lock);

      if (usingOkl) {
        return buildOKLKernelFromBinary(clInfo,
                                        kernelHash,
                                        hashDir,
                                        kernelName,
                                        launcherMetadata,
                                        deviceMetadata,
                                        kernelProps,
                                        lock);
      }

      // Regular OpenCL Kernel
      opencl::buildKernelFromProgram(clInfo,
                                     kernelName,
                                     lock);
      return new kernel(this,
                        kernelName,
                        sourceFilename,
                        clDevice,
                        clInfo.clKernel,
                        kernelProps);
    }

    modeKernel_t* device::buildOKLKernelFromBinary(const hash_t kernelHash,
                                                   const std::string &hashDir,
                                                   const std::string &kernelName,
                                                   lang::sourceMetadata_t &launcherMetadata,
                                                   lang::sourceMetadata_t &deviceMetadata,
                                                   const occa::properties &kernelProps,
                                                   io::lock_t lock) {
      info_t clInfo;
      clInfo.clDevice  = clDevice;
      clInfo.clContext = clContext;

      return buildOKLKernelFromBinary(clInfo,
                                      kernelHash,
                                      hashDir,
                                      kernelName,
                                      launcherMetadata,
                                      deviceMetadata,
                                      kernelProps,
                                      lock);
    }

    modeKernel_t* device::buildOKLKernelFromBinary(info_t &clInfo,
                                                   const hash_t kernelHash,
                                                   const std::string &hashDir,
                                                   const std::string &kernelName,
                                                   lang::sourceMetadata_t &launcherMetadata,
                                                   lang::sourceMetadata_t &deviceMetadata,
                                                   const occa::properties &kernelProps,
                                                   io::lock_t lock) {

      const std::string sourceFilename = hashDir + kc::sourceFile;
      const std::string binaryFilename = hashDir + kc::binaryFile;

      if (!clInfo.clProgram) {
        opencl::buildProgramFromBinary(clInfo,
                                       binaryFilename,
                                       kernelName,
                                       properties["compiler_flags"],
                                       lock);
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
                                       metadata.name,
                                       lock);

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
                                                const occa::properties &kernelProps) {

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
                                 const occa::properties &props) {

      if (props.get("mapped", false)) {
        return mappedAlloc(bytes, src, props);
      }

      cl_int error;

      opencl::memory *mem = new opencl::memory(this, bytes, props);

      if (src == NULL) {
        mem->clMem = clCreateBuffer(clContext,
                                    CL_MEM_READ_WRITE,
                                    bytes, NULL, &error);
        OCCA_OPENCL_ERROR("Device: clCreateBuffer", error);
      } else {
        mem->clMem = clCreateBuffer(clContext,
                                    CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                    bytes, const_cast<void*>(src), &error);
        OCCA_OPENCL_ERROR("Device: clCreateBuffer", error);

        finish();
      }

      return mem;
    }

    modeMemory_t* device::mappedAlloc(const udim_t bytes,
                                      const void *src,
                                      const occa::properties &props) {

      cl_int error;

      opencl::memory *mem = new opencl::memory(this, bytes, props);

      // Alloc pinned host buffer
      mem->clMem = clCreateBuffer(clContext,
                                  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                  bytes,
                                  NULL, &error);

      OCCA_OPENCL_ERROR("Device: clCreateBuffer", error);

      if (src != NULL){
        mem->copyFrom(src, mem->size);
      }

      // Map memory to read/write
      mem->mappedPtr = clEnqueueMapBuffer(getCommandQueue(),
                                          mem->clMem,
                                          CL_TRUE,
                                          CL_MAP_READ | CL_MAP_WRITE,
                                          0, bytes,
                                          0, NULL, NULL,
                                          &error);

      OCCA_OPENCL_ERROR("Device: clEnqueueMapBuffer", error);

      // Sync memory mapping
      finish();

      return mem;
    }

    udim_t device::memorySize() const {
      return opencl::getDeviceMemorySize(clDevice);
    }
    //==================================
  }
}
