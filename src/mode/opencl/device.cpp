#include <occa/defines.hpp>

#if OCCA_OPENCL_ENABLED

#include <occa/core/base.hpp>
#include <occa/tools/env.hpp>
#include <occa/tools/sys.hpp>
#include <occa/mode/opencl/device.hpp>
#include <occa/mode/opencl/kernel.hpp>
#include <occa/mode/opencl/memory.hpp>
#include <occa/mode/opencl/stream.hpp>
#include <occa/mode/opencl/streamTag.hpp>
#include <occa/mode/opencl/utils.hpp>
#include <occa/mode/serial/device.hpp>
#include <occa/mode/serial/kernel.hpp>
#include <occa/lang/kernelMetadata.hpp>
#include <occa/lang/primitive.hpp>
#include <occa/lang/mode/opencl.hpp>

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

      std::string compilerFlags;

      if (properties.has("kernel/compiler_flags")) {
        compilerFlags = (std::string) properties["kernel/compiler_flags"];
      } else if (env::var("OCCA_OPENCL_COMPILER_FLAGS").size()) {
        compilerFlags = env::var("OCCA_OPENCL_COMPILER_FLAGS");
      } else {
        compilerFlags = "-cl-opt-disable";
      }

      properties["kernel/compiler_flags"] = compilerFlags;
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
      return (
        occa::hash(props["compiler"])
        ^ props["compiler_flags"]
      );
    }

    lang::okl::withLauncher* device::createParser(const occa::properties &props) const {
      return new lang::okl::openclParser(props);
    }

    //---[ Stream ]---------------------
    modeStream_t* device::createStream(const occa::properties &props) {
      cl_int error;
      cl_command_queue commandQueue = clCreateCommandQueue(clContext,
                                                           clDevice,
                                                           CL_QUEUE_PROFILING_ENABLE,
                                                           &error);
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

      return (clStartTag->getTime() - clEndTag->getTime());
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
      lang::kernelMetadataMap &launcherMetadata,
      lang::kernelMetadataMap &deviceMetadata,
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
                                                   lang::kernelMetadataMap &launcherMetadata,
                                                   lang::kernelMetadataMap &deviceMetadata,
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
                                                   lang::kernelMetadataMap &launcherMetadata,
                                                   lang::kernelMetadataMap &deviceMetadata,
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
                                             launcherMetadata[kernelName]);
      if (!k.launcherKernel) {
        delete &k;
        return NULL;
      }

      // Find device kernels
      typedef std::map<int, lang::kernelMetadata> kernelOrderMap;
      kernelOrderMap clKernelMetadata;

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
        clKernelMetadata[number] = metadata;
      }

      kernelOrderMap::iterator oit = clKernelMetadata.begin();
      while (oit != clKernelMetadata.end()) {
        lang::kernelMetadata &metadata = oit->second;
        opencl::buildKernelFromProgram(clInfo,
                                       metadata.name,
                                       lock);

        kernel *clKernel = new kernel(this,
                                      metadata.name,
                                      sourceFilename,
                                      clDevice,
                                      clInfo.clKernel,
                                      kernelProps);
        clKernel->dontUseRefs();
        k.deviceKernels.push_back(clKernel);

        ++oit;
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

#endif
