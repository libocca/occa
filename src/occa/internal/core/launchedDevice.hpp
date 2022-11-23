#ifndef OCCA_INTERNAL_CORE_LAUNCHEDDEVICE_HEADER
#define OCCA_INTERNAL_CORE_LAUNCHEDDEVICE_HEADER

#include <vector>

#include <occa/internal/core/device.hpp>
#include <occa/internal/lang/kernelMetadata.hpp>
#include <occa/internal/lang/modes/withLauncher.hpp>
#include <occa/types/json.hpp>

namespace occa {
  typedef std::vector<lang::kernelMetadata_t> orderedKernelMetadata;

  class launchedModeDevice_t : public modeDevice_t {
   public:
    launchedModeDevice_t(const occa::json &json_);

    bool parseFile(const std::string &filename,
                   const std::string &outputFile,
                   const std::string &launcherOutputFile,
                   const occa::json &kernelProps,
                   lang::sourceMetadata_t &launcherMetadata,
                   lang::sourceMetadata_t &deviceMetadata);

    modeKernel_t* buildKernel(const std::string &filename,
                              const std::string &kernelName,
                              const hash_t kernelHash,
                              const occa::json &kernelProps) override;

    modeKernel_t* buildKernel(const std::string &filename,
                              const std::string &kernelName,
                              const hash_t kernelHash,
                              const bool usingOkl,
                              const occa::json &kernelProps);

    modeKernel_t* buildLauncherKernel(const hash_t kernelHash,
                                      const std::string &hashDir,
                                      const std::string &kernelName,
                                      lang::sourceMetadata_t sourceMetadata);

    orderedKernelMetadata getLaunchedKernelsMetadata(
      const std::string &kernelName,
      lang::sourceMetadata_t &deviceMetadata
    );

    //---[ Virtual Methods ]------------
    virtual lang::okl::withLauncher* createParser(const occa::json &props) const = 0;

    virtual modeKernel_t* buildKernelFromProcessedSource(
      const hash_t kernelHash,
      const std::string &hashDir,
      const std::string &kernelName,
      const std::string &sourceFilename,
      const std::string &binaryFilename,
      const bool usingOkl,
      lang::sourceMetadata_t &launcherMetadata,
      lang::sourceMetadata_t &deviceMetadata,
      const occa::json &kernelProps
    ) = 0;

    virtual modeKernel_t* buildOKLKernelFromBinary(
      const hash_t kernelHash,
      const std::string &hashDir,
      const std::string &kernelName,
      const std::string &sourceFilename,
      const std::string &binaryFilename,
      lang::sourceMetadata_t &launcherMetadata,
      lang::sourceMetadata_t &deviceMetadata,
      const occa::json &kernelProps
    ) = 0;
    //==================================
  };
}

#endif
