#include <occa/defines.hpp>

#if OCCA_METAL_ENABLED

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

      // Get device handle
    }

    device::~device() {
      // Free device
    }

    void device::finish() const {
      // Synchronize with host
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
      return new stream(this, props, NULL);
    }

    occa::streamTag device::tagStream() {
      return new occa::metal::streamTag(this, NULL);
    }

    void device::waitFor(occa::streamTag tag) {
      // Wait for event
    }

    double device::timeBetween(const occa::streamTag &startTag,
                               const occa::streamTag &endTag) {
      // ¯\_(ツ)_/¯
      return 0;
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
      // Build Metal program
      std::string source = io::read(sourceFilename, true);

      // metal::buildProgramFromSource
      // metal::saveProgramBinary
      if (usingOkl) {
        // return buildOKLKernelFromBinary
      }

      // Regular Metal Kernel
      // metal::buildKernelFromProgram

      return NULL;
    }

    modeKernel_t* device::buildOKLKernelFromBinary(const hash_t kernelHash,
                                                   const std::string &hashDir,
                                                   const std::string &kernelName,
                                                   lang::kernelMetadataMap &launcherMetadata,
                                                   lang::kernelMetadataMap &deviceMetadata,
                                                   const occa::properties &kernelProps,
                                                   io::lock_t lock) {
      // return buildOKLKernelFromBinary
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
      const std::string sourceFilename = hashDir + kc::sourceFile;
      const std::string binaryFilename = hashDir + kc::binaryFile;

      // Create wrapper kernel and set launcherKernel
      kernel &k = *(new kernel(this,
                               kernelName,
                               sourceFilename,
                               kernelProps));

      /*
      k.launcherKernel = buildLauncherKernel(kernelHash,
                                             hashDir,
                                             kernelName,
                                             launcherMetadata[kernelName]);
      */

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
        /*
        metal::buildKernelFromProgram(clInfo,
                                      metadata.name,
                                      lock);

        kernel *metalKernel = new kernel(this,
                                         metadata.name,
                                         sourceFilename,
                                         clDevice,
                                         clInfo.clKernel,
                                         kernelProps);
        metalKernel->dontUseRefs();
        metalKernel->metadata = metadata;
        k.deviceKernels.push_back(meetalKernel);
        */
      }

      return &k;
    }

    modeKernel_t* device::buildKernelFromBinary(const std::string &filename,
                                                const std::string &kernelName,
                                                const occa::properties &kernelProps) {
      return NULL;
    }
    //==================================

    //---[ Memory ]---------------------
    modeMemory_t* device::malloc(const udim_t bytes,
                                 const void *src,
                                 const occa::properties &props) {
      return NULL;
    }

    modeMemory_t* device::mappedAlloc(const udim_t bytes,
                                      const void *src,
                                      const occa::properties &props) {
      return NULL;
    }

    udim_t device::memorySize() const {
      return 0;
    }
    //==================================
  }
}

#endif
