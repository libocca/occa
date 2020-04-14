#include <occa/defines.hpp>

#ifndef OCCA_MODES_CUDA_DEVICE_HEADER
#define OCCA_MODES_CUDA_DEVICE_HEADER

#include <occa/core/launchedDevice.hpp>
#include <occa/modes/cuda/polyfill.hpp>

namespace occa {
  namespace cuda {
    class kernel;
    class memory;

    class device : public occa::launchedModeDevice_t {
      friend class kernel;

    private:
      mutable hash_t hash_;

      // We can't pass null, so we reuse a 1-byte buffer instead
      memory *nullPtr;

    public:
      int archMajorVersion, archMinorVersion;
      bool p2pEnabled;

      CUdevice  cuDevice;
      CUcontext cuContext;

      device(const occa::properties &properties_);
      virtual ~device();

      virtual void finish() const;

      virtual bool hasSeparateMemorySpace() const;

      virtual hash_t hash() const;

      virtual hash_t kernelHash(const occa::properties &props) const;

      virtual lang::okl::withLauncher* createParser(const occa::properties &props) const;

      void* getNullPtr();

      //---[ Stream ]-------------------
      virtual modeStream_t* createStream(const occa::properties &props);

      virtual streamTag tagStream();
      virtual void waitFor(streamTag tag);
      virtual double timeBetween(const streamTag &startTag,
                                 const streamTag &endTag);

      CUstream& getCuStream() const;
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
                                                   const occa::properties &kernelProps,
                                                   io::lock_t lock);

      void setArchCompilerFlags(const occa::properties &kernelProps,
                                std::string &compilerFlags);

      void compileKernel(const std::string &hashDir,
                         const std::string &kernelName,
                         const occa::properties &kernelProps,
                         io::lock_t &lock);

      modeKernel_t* buildOKLKernelFromBinary(const hash_t kernelHash,
                                             const std::string &hashDir,
                                             const std::string &kernelName,
                                             lang::sourceMetadata_t &launcherMetadata,
                                             lang::sourceMetadata_t &deviceMetadata,
                                             const occa::properties &kernelProps,
                                             io::lock_t lock);

      virtual modeKernel_t* buildKernelFromBinary(const std::string &filename,
                                                  const std::string &kernelName,
                                                  const occa::properties &props);
      //================================

      //---[ Memory ]-------------------
      virtual modeMemory_t* malloc(const udim_t bytes,
                                   const void *src,
                                   const occa::properties &props);

      virtual modeMemory_t* mappedAlloc(const udim_t bytes,
                                        const void *src,
                                        const occa::properties &props);

      modeMemory_t* unifiedAlloc(const udim_t bytes,
                                 const void *src,
                                 const occa::properties &props);

      virtual udim_t memorySize() const;
      //================================
    };
  }
}

#endif
