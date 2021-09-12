#ifndef OCCA_MODES_DPCPP_DEVICE_HEADER
#define OCCA_MODES_DPCPP_DEVICE_HEADER

#include <occa/internal/core/launchedDevice.hpp>
#include <occa/internal/modes/dpcpp/polyfill.hpp>
#include <occa/internal/modes/dpcpp/stream.hpp>


namespace occa
{
  namespace dpcpp
  {

    class device : public occa::launchedModeDevice_t
    {
    private:
      mutable hash_t hash_;

    public:
      int platformID{-1}, deviceID{-1};

      ::sycl::device dpcppDevice;
      ::sycl::context dpcppContext;

      device(const occa::json &properties_);
      virtual ~device() = default;

      virtual void finish() const override;

      virtual inline bool hasSeparateMemorySpace() const override { return true; }

      virtual hash_t hash() const override;

      virtual hash_t kernelHash(const occa::json &props) const override;

      virtual lang::okl::withLauncher *createParser(const occa::json &props) const override;

      //---[ Stream ]-------------------
      virtual modeStream_t *createStream(const occa::json &props) override;

      virtual occa::streamTag tagStream() override;
      virtual void waitFor(occa::streamTag tag) override;
      virtual double timeBetween(const occa::streamTag &startTag,
                                 const occa::streamTag &endTag) override;
      //================================

      //---[ Kernel ]-------------------
      virtual modeKernel_t *buildKernelFromProcessedSource(const hash_t kernelHash,
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
                         const occa::json &kernelProps);

      virtual modeKernel_t *buildOKLKernelFromBinary(const hash_t kernelHash,
                                             const std::string &hashDir,
                                             const std::string &kernelName,
                                             const std::string &sourceFilename,
                                             const std::string &binaryFilename,
                                             lang::sourceMetadata_t &launcherMetadata,
                                             lang::sourceMetadata_t &deviceMetadata,
                                             const occa::json &kernelProps) override;

      virtual modeKernel_t *buildKernelFromBinary(const std::string &filename,
                                                  const std::string &kernelName,
                                                  const occa::json &props) override;

      int maxDims() const;
      dim maxOuterDims() const;
      dim maxInnerDims() const;
      udim_t maxInnerSize() const;
      //================================

      //---[ Memory ]-------------------
      virtual modeMemory_t *malloc(const udim_t bytes,
                                   const void *src,
                                   const occa::json &props) override;

      virtual modeMemory_t *wrapMemory(const void *ptr,
                               const udim_t bytes,
                               const occa::json &props) override;

      virtual udim_t memorySize() const override;
      //================================
    };
  } // namespace dpcpp
} // namespace occa

#endif
