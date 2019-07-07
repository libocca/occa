#include <occa/core/base.hpp>
#include <occa/tools/env.hpp>
#include <occa/tools/sys.hpp>
#include <occa/modes/metal/device.hpp>
#include <occa/modes/metal/kernel.hpp>
#include <occa/modes/metal/memory.hpp>
#include <occa/modes/metal/stream.hpp>
#include <occa/modes/metal/streamTag.hpp>
#include <occa/modes/serial/device.hpp>
#include <occa/modes/serial/kernel.hpp>
#include <occa/lang/modes/metal.hpp>

namespace occa {
  namespace metal {
    device::device(const occa::properties &properties_) :
      occa::launchedModeDevice_t(properties_) {

      OCCA_ERROR("[Metal] device not given a [device_id] integer",
                 properties.has("device_id") &&
                 properties["device_id"].isNumber());

      deviceID = properties.get<int>("device_id");

      metalDevice = api::metal::getDevice(deviceID);
      metalCommandQueue = metalDevice.createCommandQueue();
    }

    device::~device() {
      metalDevice.free();
    }

    void device::finish() const {
      metal::stream &stream = (
        *((metal::stream*) (currentStream.getModeStream()))
      );
      metalDevice.finish(stream.metalCommandQueue);
    }

    bool device::hasSeparateMemorySpace() const {
      return true;
    }

    hash_t device::hash() const {
      if (!hash_.initialized) {
        std::stringstream ss;
        ss << "device: " << deviceID;
        hash_ = occa::hash(ss.str());
      }
      return hash_;
    }

    hash_t device::kernelHash(const occa::properties &props) const {
      return occa::hash("metal");
    }

    lang::okl::withLauncher* device::createParser(const occa::properties &props) const {
      return new lang::okl::metalParser(props);
    }

    //---[ Stream ]---------------------
    modeStream_t* device::createStream(const occa::properties &props) {
      return new stream(this, props, metalCommandQueue);
    }

    occa::streamTag device::tagStream() {
      return new occa::metal::streamTag(this, metalDevice.createEvent());
    }

    void device::waitFor(occa::streamTag tag) {
      occa::metal::streamTag *metalTag = (
        dynamic_cast<occa::metal::streamTag*>(tag.getModeStreamTag())
      );
      metalDevice.waitFor(metalTag->metalEvent);
    }

    double device::timeBetween(const occa::streamTag &startTag,
                               const occa::streamTag &endTag) {
      occa::metal::streamTag *metalStartTag = (
        dynamic_cast<occa::metal::streamTag*>(startTag.getModeStreamTag())
      );
      occa::metal::streamTag *metalEndTag = (
        dynamic_cast<occa::metal::streamTag*>(endTag.getModeStreamTag())
      );

      waitFor(endTag);

      return (metalEndTag->getTime() - metalStartTag->getTime());
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
      OCCA_ERROR("Metal kernels need to use OKL for now",
                 usingOkl);

      std::string source = io::read(sourceFilename, true);

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
      orderedKernelMetadata launchedKernelsMetadata = getLaunchedKernelsMetadata(
        kernelName,
        deviceMetadata
      );

      const int launchedKernelsCount = (int) launchedKernelsMetadata.size();
      for (int i = 0; i < launchedKernelsCount; ++i) {
        lang::kernelMetadata &metadata = launchedKernelsMetadata[i];

        api::metal::kernel_t metalKernel = metalDevice.buildKernel(source,
                                                                   metadata.name,
                                                                   lock);
        kernel *deviceKernel = new kernel(this,
                                          metadata.name,
                                          sourceFilename,
                                          metalDevice,
                                          metalKernel,
                                          kernelProps);
        deviceKernel->dontUseRefs();
        deviceKernel->metadata = metadata;
        k.deviceKernels.push_back(deviceKernel);
      }

      return &k;
    }

    modeKernel_t* device::buildOKLKernelFromBinary(const hash_t kernelHash,
                                                   const std::string &hashDir,
                                                   const std::string &kernelName,
                                                   lang::kernelMetadataMap &launcherMetadata,
                                                   lang::kernelMetadataMap &deviceMetadata,
                                                   const occa::properties &kernelProps,
                                                   io::lock_t lock) {
      OCCA_FORCE_ERROR("Metal does not support building from binary");
      return NULL;
    }

    modeKernel_t* device::buildOKLKernelFromBinary(info_t &clInfo,
                                                   const hash_t kernelHash,
                                                   const std::string &hashDir,
                                                   const std::string &kernelName,
                                                   lang::kernelMetadataMap &launcherMetadata,
                                                   lang::kernelMetadataMap &deviceMetadata,
                                                   const occa::properties &kernelProps,
                                                   io::lock_t lock) {
      OCCA_FORCE_ERROR("Metal does not support building from binary");
      return NULL;
    }

    modeKernel_t* device::buildKernelFromBinary(const std::string &filename,
                                                const std::string &kernelName,
                                                const occa::properties &kernelProps) {
      OCCA_FORCE_ERROR("Metal does not support building from binary");
      return NULL;
    }
    //==================================

    //---[ Memory ]---------------------
    modeMemory_t* device::malloc(const udim_t bytes,
                                 const void *src,
                                 const occa::properties &props) {
      metal::memory *mem = new metal::memory(this, bytes, props);

      mem->metalBuffer = metalDevice.malloc(bytes, src);

      return mem;
    }

    udim_t device::memorySize() const {
      return metalDevice.getMemorySize();
    }
    //==================================
  }
}
