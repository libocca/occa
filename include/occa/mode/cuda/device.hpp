#if OCCA_CUDA_ENABLED
#  ifndef OCCA_MODES_CUDA_DEVICE_HEADER
#  define OCCA_MODES_CUDA_DEVICE_HEADER

#include <occa/defines.hpp>
#include <occa/core/device.hpp>

#include <cuda.h>

namespace occa {
  namespace cuda {
    class kernel;

    class device : public occa::modeDevice_t {
      friend class kernel;

    private:
      mutable hash_t hash_;

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

      //---[ Stream ]-------------------
      virtual modeStream_t* createStream(const occa::properties &props);

      virtual streamTag tagStream();
      virtual void waitFor(streamTag tag);
      virtual double timeBetween(const streamTag &startTag,
                                 const streamTag &endTag);

      CUstream& getCuStream() const;
      //================================

      //---[ Kernel ]-------------------
      bool parseFile(const std::string &filename,
                     const std::string &outputFile,
                     const std::string &hostOutputFile,
                     const occa::properties &kernelProps,
                     lang::kernelMetadataMap &hostMetadata,
                     lang::kernelMetadataMap &deviceMetadata);

      virtual modeKernel_t* buildKernel(const std::string &filename,
                                        const std::string &kernelName,
                                        const hash_t kernelHash,
                                        const occa::properties &props);

      void setArchCompilerFlags(occa::properties &kernelProps);

      void compileKernel(const std::string &hashDir,
                         const std::string &kernelName,
                         const occa::properties &kernelProps,
                         io::lock_t &lock);

      modeKernel_t* buildOKLKernelFromBinary(const std::string &hashDir,
                                             const std::string &kernelName,
                                             lang::kernelMetadataMap &hostMetadata,
                                             lang::kernelMetadataMap &deviceMetadata,
                                             const occa::properties &kernelProps,
                                             const io::lock_t &lock = io::lock_t());

      modeKernel_t* buildLauncherKernel(const std::string &hashDir,
                                        const std::string &kernelName,
                                        lang::kernelMetadata &hostMetadata);

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

#  endif
#endif
