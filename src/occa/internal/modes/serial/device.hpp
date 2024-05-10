#ifndef OCCA_INTERNAL_MODES_SERIAL_DEVICE_HEADER
#define OCCA_INTERNAL_MODES_SERIAL_DEVICE_HEADER

#include <occa/defines.hpp>
#include <occa/internal/core/device.hpp>

namespace occa {
  namespace serial {
    class device : public occa::modeDevice_t {
      mutable hash_t hash_;

    public:
      device(const occa::json &properties_);
      virtual ~device() = default;

      bool hasSeparateMemorySpace() const override;

      hash_t hash() const override;

      hash_t kernelHash(const occa::json &props) const override;

      //---[ Stream ]-------------------
      modeStream_t* createStream(const occa::json &props) override;
      modeStream_t* wrapStream(void* ptr, const occa::json &props) override;

      streamTag tagStream() override;
      void waitFor(streamTag tag) override;
      double timeBetween(const streamTag &startTag,
                         const streamTag &endTag) override;
      //================================

      //---[ Kernel ]-------------------
      virtual bool parseFile(const std::string &filename,
                             const std::string &outputFile,
                             const occa::json &kernelProps,
                             lang::sourceMetadata_t &metadata);

      modeKernel_t* buildKernel(const std::string &filename,
                                const std::string &kernelName,
                                const hash_t kernelHash,
                                const occa::json &kernelProps) override;

      modeKernel_t* buildLauncherKernel(const std::string &filename,
                                        const std::string &kernelName,
                                        const hash_t kernelHash);

      modeKernel_t* buildKernel(const std::string &filename,
                                const std::string &kernelName,
                                const hash_t kernelHash,
                                const occa::json &kernelProps,
                                const bool isLauncherKernel);

      modeKernel_t* buildKernelFromBinary(const std::string &filename,
                                          const std::string &kernelName,
                                          const occa::json &kernelProps) override;

      virtual modeKernel_t* buildKernelFromBinary(const std::string &filename,
                                                  const std::string &kernelName,
                                                  const occa::json &kernelProps,
                                                  lang::kernelMetadata_t &metadata);
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
