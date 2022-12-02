#ifndef OCCA_INTERNAL_MODES_METAL_DEVICE_HEADER
#define OCCA_INTERNAL_MODES_METAL_DEVICE_HEADER

#include <occa/internal/core/launchedDevice.hpp>
#include <occa/internal/api/metal.hpp>

namespace occa {
  namespace metal {
    class info_t;

    class device : public occa::launchedModeDevice_t {

    private:
      mutable hash_t hash_;

    public:
      int deviceID;

      api::metal::device_t metalDevice;
      api::metal::commandQueue_t metalCommandQueue;

      device(const occa::json &properties_);
      virtual ~device();

      bool hasSeparateMemorySpace() const override;

      hash_t hash() const override;

      hash_t kernelHash(const occa::json &props) const override;

      lang::okl::withLauncher* createParser(const occa::json &props) const override;

      //---[ Stream ]-------------------
      modeStream_t* createStream(const occa::json &props) override;
      modeStream_t* wrapStream(void* ptr, const occa::json &props) override;

      streamTag tagStream() override;
      void waitFor(streamTag tag) override;
      double timeBetween(const streamTag &startTag,
                         const streamTag &endTag) override;
      //================================

      //---[ Kernel ]-------------------
      modeKernel_t* buildKernelFromProcessedSource(const hash_t kernelHash,
                                                   const std::string &hashDir,
                                                   const std::string &kernelName,
                                                   const std::string &sourceFilename,
                                                   const std::string &binaryFilename,
                                                   const bool usingOkl,
                                                   lang::sourceMetadata_t &launcherMetadata,
                                                   lang::sourceMetadata_t &deviceMetadata,
                                                   const occa::json &kernelProps) override;

      void compileKernel(const std::string &hashDir,
                         const std::string &kernelName,
                         const std::string &sourceFilename,
                         const std::string &binaryFilename,
                         const occa::json &kernelProps);

      modeKernel_t* buildOKLKernelFromBinary(const hash_t kernelHash,
                                             const std::string &hashDir,
                                             const std::string &kernelName,
                                             const std::string &sourceFilename,
                                             const std::string &binaryFilename,
                                             lang::sourceMetadata_t &launcherMetadata,
                                             lang::sourceMetadata_t &deviceMetadata,
                                             const occa::json &kernelProps) override;

      modeKernel_t* buildKernelFromBinary(const std::string &filename,
                                          const std::string &kernelName,
                                          const occa::json &kernelProps) override;
      //================================

      //---[ Memory ]-------------------
      modeMemory_t* malloc(const udim_t bytes,
                           const void *src,
                           const occa::json &props) override;

      modeMemory_t* wrapMemory(const void *ptr,
                               const udim_t bytes,
                               const occa::json &props) override;

      modeMemoryPool_t* createMemoryPool(const occa::json &props) override;

      udim_t memorySize() const override;
      //================================

      void* unwrap() override;
    };
  }
}

#endif
