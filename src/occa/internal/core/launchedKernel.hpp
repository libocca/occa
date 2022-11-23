#ifndef OCCA_INTERNAL_CORE_LAUNCHEDKERNEL_HEADER
#define OCCA_INTERNAL_CORE_LAUNCHEDKERNEL_HEADER

#include <occa/internal/core/kernel.hpp>

namespace occa {
  class launchedModeDevice_t;

  class launchedModeKernel_t : public modeKernel_t {
    friend class launchedModeDevice_t;

   protected:
    occa::modeKernel_t *launcherKernel;
    std::vector<modeKernel_t*> deviceKernels;

   public:
    launchedModeKernel_t(modeDevice_t *modeDevice_,
                         const std::string &name_,
                         const std::string &sourceFilename_,
                         const occa::json &properties_);

    virtual ~launchedModeKernel_t();

    const lang::kernelMetadata_t& getMetadata() const override;

    void run() const override;
    void launcherRun() const;

    //---[ Virtual Methods ]------------
    virtual void deviceRun() const = 0;
    //==================================
  };
}

#endif
