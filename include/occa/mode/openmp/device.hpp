#include <occa/defines.hpp>

#if OCCA_OPENMP_ENABLED
#  ifndef OCCA_MODES_OPENMP_DEVICE_HEADER
#  define OCCA_MODES_OPENMP_DEVICE_HEADER

#include <occa/mode/serial/device.hpp>

namespace occa {
  namespace openmp {
    class device : public serial::device {
    private:
      // Make sure we don't warn everytime we switch from [OpenMP] -> [Serial]
      //   due to compiler changes
      std::string lastCompiler;
      std::string lastCompilerOpenMPFlag;

    public:
      device(const occa::properties &properties_);

      virtual hash_t kernelHash(const occa::properties &props) const;

      virtual bool parseFile(const std::string &filename,
                             const std::string &outputFile,
                             const occa::properties &kernelProps,
                             lang::kernelMetadataMap &metadata);

      virtual modeKernel_t* buildKernel(const std::string &filename,
                                        const std::string &kernelName,
                                        const hash_t kernelHash,
                                        const occa::properties &kernelProps);
    };
  }
}

#  endif
#endif
