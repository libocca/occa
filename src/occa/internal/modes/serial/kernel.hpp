#ifndef OCCA_INTERNAL_MODES_SERIAL_KERNEL_HEADER
#define OCCA_INTERNAL_MODES_SERIAL_KERNEL_HEADER

#include <vector>

#include <occa/defines.hpp>
#include <occa/internal/core/kernel.hpp>
#include <occa/internal/utils/sys.hpp>

namespace occa {
  namespace serial {
    class device;

    class kernel : public occa::modeKernel_t {
    protected:
      void *dlHandle;
      functionPtr_t function;
      mutable std::vector<void*> vArgs;

    public:
      bool isLauncherKernel;

      kernel(modeDevice_t *modeDevice_,
             const std::string &name_,
             const std::string &sourceFilename_,
             const occa::json &properties_);
      virtual ~kernel();

      int maxDims() const override;
      dim maxOuterDims() const override;
      dim maxInnerDims() const override;

      const lang::kernelMetadata_t& getMetadata() const override;

      void run() const override;

      friend class device;
    };
  }
}
#endif
