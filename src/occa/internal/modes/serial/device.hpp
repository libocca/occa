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
      virtual ~device();

      virtual void finish() const;

      virtual bool hasSeparateMemorySpace() const;

      virtual hash_t hash() const;

      virtual hash_t kernelHash(const occa::json &props) const;

      //---[ Stream ]-------------------
      virtual modeStream_t* createStream(const occa::json &props);

      virtual streamTag tagStream();
      virtual void waitFor(streamTag tag);
      virtual double timeBetween(const streamTag &startTag,
                                 const streamTag &endTag);
      //================================

      //---[ Kernel ]-------------------
      virtual bool parseFile(const std::string &filename,
                             const std::string &outputFile,
                             const occa::json &kernelProps,
                             lang::sourceMetadata_t &metadata);

      virtual modeKernel_t* buildKernel(const std::string &filename,
                                        const std::string &kernelName,
                                        const hash_t kernelHash,
                                        const occa::json &kernelProps);

      modeKernel_t* buildLauncherKernel(const std::string &filename,
                                        const std::string &kernelName,
                                        const hash_t kernelHash);

      modeKernel_t* buildKernel(const std::string &filename,
                                const std::string &kernelName,
                                const hash_t kernelHash,
                                const occa::json &kernelProps,
                                const bool isLauncerKernel);

      virtual modeKernel_t* buildKernelFromBinary(const std::string &filename,
                                                  const std::string &kernelName,
                                                  const occa::json &kernelProps);

      virtual modeKernel_t* buildKernelFromBinary(const std::string &filename,
                                                  const std::string &kernelName,
                                                  const occa::json &kernelProps,
                                                  lang::kernelMetadata_t &metadata);
      //================================

      //---[ Memory ]-------------------
      virtual modeMemory_t* malloc(const udim_t bytes,
                                   const void *src,
                                   const occa::json &props);

      modeMemory_t* wrapMemory(const void *ptr,
                               const udim_t bytes,
                               const occa::json &props);

      virtual udim_t memorySize() const;
      //================================
    };
  }
}

#endif
