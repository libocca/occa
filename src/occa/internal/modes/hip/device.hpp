#ifndef OCCA_INTERNAL_MODES_HIP_DEVICE_HEADER
#define OCCA_INTERNAL_MODES_HIP_DEVICE_HEADER

#include <occa/internal/core/launchedDevice.hpp>
#include <occa/internal/modes/hip/polyfill.hpp>


namespace occa {
  namespace hip {
    class kernel;

    class device : public occa::launchedModeDevice_t {
      friend class kernel;

    private:
      mutable hash_t hash_;

    public:
      int archMajorVersion, archMinorVersion;
      bool p2pEnabled;

      hipDevice_t hipDevice;
      int deviceID;

      device(const occa::json &properties_);
      virtual ~device();

      bool hasSeparateMemorySpace() const override;

      hash_t hash() const override;

      hash_t kernelHash(const occa::json &props) const override;

      lang::okl::withLauncher* createParser(const occa::json &props) const override;

      void getDeviceArchVersion(int *archMajorVersion_,
                                int *archMinorVersion_) const override;

      //---[ Stream ]-------------------
      modeStream_t* createStream(const occa::json &props) override;
      modeStream_t* wrapStream(void* ptr, const occa::json &props) override;

      streamTag tagStream() override;
      void waitFor(streamTag tag) override;
      double timeBetween(const streamTag &startTag,
                         const streamTag &endTag) override;

      hipStream_t& getHipStream() const;
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

      void setArchCompilerFlags(occa::json &kernelProps);

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
                                          const occa::json &props) override;
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
