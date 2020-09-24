#include <occa/core/base.hpp>
#include <occa/tools/env.hpp>
#include <occa/tools/sys.hpp>
#include <occa/modes/dpcpp/device.hpp>
#include <occa/modes/dpcpp/kernel.hpp>
#include <occa/modes/dpcpp/memory.hpp>
#include <occa/modes/dpcpp/stream.hpp>
#include <occa/modes/dpcpp/streamTag.hpp>
#include <occa/modes/dpcpp/utils.hpp>
#include <occa/modes/serial/device.hpp>
#include <occa/modes/serial/kernel.hpp>
#include <occa/lang/kernelMetadata.hpp>
#include <occa/lang/primitive.hpp>
#include <occa/lang/modes/dpcpp.hpp>

namespace occa {
  namespace dpcpp {
    device::device(const occa::properties &properties_) :
      occa::launchedModeDevice_t(properties_) {

      if (!properties.has("wrapped")) {
        int error;
        OCCA_ERROR("[DPCPP] device not given a [platform_id] integer",
                   properties.has("platform_id") &&
                   properties["platform_id"].isNumber());

        OCCA_ERROR("[DPCPP] device not given a [device_id] integer",
                   properties.has("device_id") &&
                   properties["device_id"].isNumber());

      /*  platformID = properties.get<int>("platform_id");
        deviceID   = properties.get<int>("device_selector");
*/
        dpcppDevice = ::sycl::device();

//        dpcppContext = clCreateContext(NULL, 1, &clDevice, NULL, NULL, &error);
//        OCCA_DPCPP_ERROR("Device: Creating Context", error);
      }

      occa::json &kernelProps = properties["kernel"];
      std::string compilerFlags;

      // Use "-cl-opt-disable" for debug-mode
      if (env::var("OCCA_DPCPP_COMPILER_FLAGS").size()) {
        compilerFlags = env::var("OCCA_DPCPP_COMPILER_FLAGS");
      } else if (kernelProps.has("compiler_flags")) {
        compilerFlags = (std::string) kernelProps["compiler_flags"];
      }

      kernelProps["compiler_flags"] = compilerFlags;
    }

    device::~device() {
    }

    void device::finish() const {
      getCommandQueue()->wait();
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
      return new lang::okl::dpcppParser(props);
    }

    //---[ Stream ]---------------------
    modeStream_t* device::createStream(const occa::properties &props) {
      ::sycl::queue* q = new ::sycl::queue();
      return new stream(this, props, q);
    }

    occa::streamTag device::tagStream() {
      return new occa::dpcpp::streamTag(this);
    }

    void device::waitFor(occa::streamTag tag) {
/*      occa::opencl::streamTag *clTag = (
        dynamic_cast<occa::opencl::streamTag*>(tag.getModeStreamTag())
      );
      OCCA_OPENCL_ERROR("Device: Waiting For Tag",
                        clWaitForEvents(1, &(clTag->clEvent)));*/
    }

    double device::timeBetween(const occa::streamTag &startTag,
                               const occa::streamTag &endTag) {
      occa::dpcpp::streamTag *dpcppStartTag = (
        dynamic_cast<occa::dpcpp::streamTag*>(startTag.getModeStreamTag())
      );
      occa::dpcpp::streamTag *dpcppEndTag = (
        dynamic_cast<occa::dpcpp::streamTag*>(endTag.getModeStreamTag())
      );

      finish();

      return (dpcppEndTag->getTime() - dpcppStartTag->getTime());
    }

    ::sycl::queue *device::getCommandQueue() const {
      occa::dpcpp::stream *stream = (occa::dpcpp::stream*) currentStream.getModeStream();
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
/*      info_t clInfo;
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
*/
	return NULL;    
  }

    modeKernel_t* device::buildOKLKernelFromBinary(const hash_t kernelHash,
                                                   const std::string &hashDir,
                                                   const std::string &kernelName,
                                                   lang::sourceMetadata_t &launcherMetadata,
                                                   lang::sourceMetadata_t &deviceMetadata,
                                                   const occa::properties &kernelProps,
                                                   io::lock_t lock) {
/*      info_t clInfo;
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
*/
 	return NULL;
    }

    modeKernel_t* device::buildOKLKernelFromBinary(info_t &clInfo,
                                                   const hash_t kernelHash,
                                                   const std::string &hashDir,
                                                   const std::string &kernelName,
                                                   lang::sourceMetadata_t &launcherMetadata,
                                                   lang::sourceMetadata_t &deviceMetadata,
                                                   const occa::properties &kernelProps,
                                                   io::lock_t lock) {

/*      const std::string sourceFilename = hashDir + kc::sourceFile;
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
      */
	    return NULL;
    }

    modeKernel_t* device::buildKernelFromBinary(const std::string &filename,
                                                const std::string &kernelName,
                                                const occa::properties &kernelProps) {

/*      info_t clInfo;
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
*/
	    return NULL;
		    
	}
    //==================================

    //---[ Memory ]---------------------
    modeMemory_t* device::malloc(const udim_t bytes,
                                 const void *src,
                                 const occa::properties &props) {

      ::sycl::queue* q = getCommandQueue();
/*      if (props.get("mapped", false)) {
        return mappedAlloc(bytes, src, props);
      }
*/

      memory *mem = new memory(this, bytes, props);

      if (src == NULL) {
        mem->dpcppMem = malloc_device(bytes, *q);
      } else {
	mem->dpcppMem = malloc_device(bytes, *q);
	q->memcpy(mem->dpcppMem, src, bytes);

        finish();
      }

      mem->rootDpcppMem = &mem->dpcppMem;

      return mem;
    }

    modeMemory_t* device::mappedAlloc(const udim_t bytes,
                                      const void *src,
                                      const occa::properties &props) {

/*
      opencl::memory *mem = new opencl::memory(this, bytes, props);

      // Alloc pinned host buffer
      mem->clMem = clCreateBuffer(clContext,
                                  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                  bytes,
                                  NULL, &error);
      mem->rootClMem = &mem->clMem;

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
*/
	    return NULL;
      }

    udim_t device::memorySize() const {
      return dpcpp::getDeviceMemorySize(dpcppDevice);
    }
    //==================================
  }
}
