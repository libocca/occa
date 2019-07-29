#ifndef OCCA_MODES_SERIAL_DEVICE_HEADER
#define OCCA_MODES_SERIAL_DEVICE_HEADER

#include <occa/defines.hpp>
#include <occa/core/device.hpp>

namespace occa {
  namespace serial {
    class device : public occa::modeDevice_t {
      mutable hash_t hash_;

    public:
      device(const occa::properties &properties_);
      virtual ~device();

      virtual void finish() const;

      virtual bool hasSeparateMemorySpace() const;

      virtual hash_t hash() const;

      virtual hash_t kernelHash(const occa::properties &props) const;

      //---[ Stream ]-------------------
      virtual modeStream_t* createStream(const occa::properties &props);

      virtual streamTag tagStream();
      virtual void waitFor(streamTag tag);
      virtual double timeBetween(const streamTag &startTag,
                                 const streamTag &endTag);
      //================================

      //---[ Kernel ]-------------------
      virtual bool parseFile(const std::string &filename,
                             const std::string &outputFile,
                             const occa::properties &kernelProps,
                             lang::sourceMetadata_t &metadata);

      virtual modeKernel_t* buildKernel(const std::string &filename,
                                        const std::string &kernelName,
                                        const hash_t kernelHash,
                                        const occa::properties &kernelProps);

      modeKernel_t* buildLauncherKernel(const std::string &filename,
                                        const std::string &kernelName,
                                        const hash_t kernelHash);

      modeKernel_t* buildKernel(const std::string &filename,
                                const std::string &kernelName,
                                const hash_t kernelHash,
                                const occa::properties &kernelProps,
                                const bool isLauncerKernel);

      virtual modeKernel_t* buildKernelFromBinary(const std::string &filename,
                                                  const std::string &kernelName,
                                                  const occa::properties &kernelProps);

      virtual modeKernel_t* buildKernelFromBinary(const std::string &filename,
                                                  const std::string &kernelName,
                                                  const occa::properties &kernelProps,
                                                  lang::kernelMetadata_t &metadata);
      //================================

      //---[ Memory ]-------------------
      virtual modeMemory_t* malloc(const udim_t bytes,
                                   const void *src,
                                   const occa::properties &props);

      virtual udim_t memorySize() const;
      //================================
    };
  }
}

#endif
