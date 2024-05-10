#ifndef OCCA_INTERNAL_MODES_OPENMP_DEVICE_HEADER
#define OCCA_INTERNAL_MODES_OPENMP_DEVICE_HEADER

#include <occa/internal/modes/serial/device.hpp>

namespace occa {
  namespace openmp {
    class device : public serial::device {
    private:
      // Make sure we don't warn everytime we switch from [OpenMP] -> [Serial]
      //   due to compiler changes
      std::string lastCompiler;
      std::string lastCompilerOpenMPFlag;

    public:
      device(const occa::json &properties_);
      virtual ~device() = default;

      hash_t hash() const override;

      hash_t kernelHash(const occa::json &props) const override;

      bool parseFile(const std::string &filename,
                     const std::string &outputFile,
                     const occa::json &kernelProps,
                     lang::sourceMetadata_t &metadata) override;

      modeKernel_t* buildKernel(const std::string &filename,
                                const std::string &kernelName,
                                const hash_t kernelHash,
                                const occa::json &kernelProps) override;
    };
  }
}

#endif
