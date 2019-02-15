#ifndef OCCA_CORE_LAUNCHEDDEVICE_HEADER
#define OCCA_CORE_LAUNCHEDDEVICE_HEADER

#include <occa/core/device.hpp>
#include <occa/lang/mode/withLauncher.hpp>
#include <occa/tools/properties.hpp>

namespace occa {
  class launchedModeDevice_t : public modeDevice_t {
  public:
    launchedModeDevice_t(const occa::properties &properties_);

    bool parseFile(const std::string &filename,
                   const std::string &outputFile,
                   const std::string &launcherOutputFile,
                   const occa::properties &kernelProps,
                   lang::kernelMetadataMap &launcherMetadata,
                   lang::kernelMetadataMap &deviceMetadata);

    virtual modeKernel_t* buildKernel(const std::string &filename,
                                      const std::string &kernelName,
                                      const hash_t kernelHash,
                                      const occa::properties &kernelProps);

    modeKernel_t* buildKernel(const std::string &filename,
                              const std::string &kernelName,
                              const hash_t kernelHash,
                              const bool usingOkl,
                              const occa::properties &kernelProps);

    modeKernel_t* buildLauncherKernel(const hash_t kernelHash,
                                      const std::string &hashDir,
                                      const std::string &kernelName,
                                      lang::kernelMetadata &launcherMetadata);

    //---[ Virtual Methods ]------------
    virtual lang::okl::withLauncher* createParser(const occa::properties &props) const = 0;

    virtual modeKernel_t* buildKernelFromProcessedSource(
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
    ) = 0;

    virtual modeKernel_t* buildOKLKernelFromBinary(
      const hash_t kernelHash,
      const std::string &hashDir,
      const std::string &kernelName,
      lang::kernelMetadataMap &launcherMetadata,
      lang::kernelMetadataMap &deviceMetadata,
      const occa::properties &kernelProps,
      io::lock_t lock
    ) = 0;
    //==================================
  };
}

#endif
