#ifndef OCCA_CORE_LAUNCHEDKERNEL_HEADER
#define OCCA_CORE_LAUNCHEDKERNEL_HEADER

#include <occa/core/kernel.hpp>

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
                         const occa::properties &properties_);

    ~launchedModeKernel_t();

    const lang::kernelMetadata_t& getMetadata() const;

    void run() const;
    void launcherRun() const;

    //---[ Virtual Methods ]------------
    virtual void deviceRun() const = 0;
    //==================================
  };
}

#endif
