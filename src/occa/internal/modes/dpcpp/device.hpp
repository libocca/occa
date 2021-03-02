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
      int platformID, deviceID;

      ::sycl::device dpcppDevice;
      ::sycl::context dpcppContext;

      device(const occa::json &properties_);
      virtual ~device() = default;

      virtual void finish() const;

      virtual inline bool hasSeparateMemorySpace() const { return true; }

      virtual hash_t hash() const;

      virtual hash_t kernelHash(const occa::json &props) const;

      virtual lang::okl::withLauncher *createParser(const occa::json &props) const;

      //---[ Stream ]-------------------
      virtual modeStream_t *createStream(const occa::json &props);

      virtual occa::streamTag tagStream();
      virtual void waitFor(occa::streamTag tag);
      virtual double timeBetween(const occa::streamTag &startTag,
                                 const occa::streamTag &endTag);
      //================================

      //---[ Kernel ]-------------------
      modeKernel_t *buildKernelFromProcessedSource(const hash_t kernelHash,
                                                   const std::string &hashDir,
                                                   const std::string &kernelName,
                                                   const std::string &sourceFilename,
                                                   const std::string &binaryFilename,
                                                   const bool usingOkl,
                                                   lang::sourceMetadata_t &launcherMetadata,
                                                   lang::sourceMetadata_t &deviceMetadata,
                                                   const occa::json &kernelProps,
                                                   io::lock_t lock);
      void setArchCompilerFlags(occa::json &kernelProps);

      void compileKernel(const std::string &hashDir,
                         const std::string &kernelName,
                         const occa::json &kernelProps,
                         io::lock_t &lock);

      modeKernel_t *buildOKLKernelFromBinary(const hash_t kernelHash,
                                             const std::string &hashDir,
                                             const std::string &kernelName,
                                             lang::sourceMetadata_t &launcherMetadata,
                                             lang::sourceMetadata_t &deviceMetadata,
                                             const occa::json &kernelProps,
                                             io::lock_t lock);

      virtual modeKernel_t *buildKernelFromBinary(const std::string &filename,
                                                  const std::string &kernelName,
                                                  const occa::json &props);

      int maxDims() const;
      dim maxOuterDims() const;
      dim maxInnerDims() const;
      udim_t maxInnerSize() const;
      //================================

      //---[ Memory ]-------------------
      virtual modeMemory_t *malloc(const udim_t bytes,
                                   const void *src,
                                   const occa::json &props);

      virtual modeMemory_t *mappedAlloc(const udim_t bytes,
                                        const void *src,
                                        const occa::json &props);

      modeMemory_t *unifiedAlloc(const udim_t bytes,
                                 const void *src,
                                 const occa::json &props);

      modeMemory_t *wrapMemory(const void *ptr,
                               const udim_t bytes,
                               const occa::json &props);

      virtual udim_t memorySize() const;
      //================================
    };
  } // namespace dpcpp
} // namespace occa

#endif
