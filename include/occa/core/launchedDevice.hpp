#ifndef OCCA_CORE_LAUNCHEDDEVICE_HEADER
#define OCCA_CORE_LAUNCHEDDEVICE_HEADER

#include <vector>

#include <occa/core/device.hpp>
#include <occa/lang/kernelMetadata.hpp>
#include <occa/lang/modes/withLauncher.hpp>
#include <occa/tools/properties.hpp>

namespace occa {
  typedef std::vector<lang::kernelMetadata_t> orderedKernelMetadata;

  class launchedModeDevice_t : public modeDevice_t {
  public:
    launchedModeDevice_t(const occa::properties &properties_);

    bool parseFile(const std::string &filename,
                   const std::string &outputFile,
                   const std::string &launcherOutputFile,
                   const occa::properties &kernelProps,
                   lang::sourceMetadata_t &launcherMetadata,
                   lang::sourceMetadata_t &deviceMetadata);

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
                                      lang::sourceMetadata_t sourceMetadata);

    orderedKernelMetadata getLaunchedKernelsMetadata(
      const std::string &kernelName,
      lang::sourceMetadata_t &deviceMetadata
    );

    //---[ Virtual Methods ]------------
    virtual lang::okl::withLauncher* createParser(const occa::properties &props) const = 0;

    virtual modeKernel_t* buildKernelFromProcessedSource(
      const hash_t kernelHash,
      const std::string &hashDir,
      const std::string &kernelName,
      const std::string &sourceFilename,
      const std::string &binaryFilename,
      const bool usingOkl,
      lang::sourceMetadata_t &launcherMetadata,
      lang::sourceMetadata_t &deviceMetadata,
      const occa::properties &kernelProps,
      io::lock_t lock
    ) = 0;

    virtual modeKernel_t* buildOKLKernelFromBinary(
      const hash_t kernelHash,
      const std::string &hashDir,
      const std::string &kernelName,
      lang::sourceMetadata_t &launcherMetadata,
      lang::sourceMetadata_t &deviceMetadata,
      const occa::properties &kernelProps,
      io::lock_t lock
    ) = 0;
    //==================================
  };
}

#endif
