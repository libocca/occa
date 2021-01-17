#ifndef OCCA_INTERNAL_CORE_KERNEL_HEADER
#define OCCA_INTERNAL_CORE_KERNEL_HEADER

#include <occa/core/kernel.hpp>
#include <occa/types/json.hpp>
#include <occa/internal/utils/gc.hpp>
#include <occa/internal/lang/kernelMetadata.hpp>

namespace occa {
  class modeKernel_t : public gc::ringEntry_t {
   public:
    // Information about the kernel
    occa::modeDevice_t *modeDevice;
    std::string name;
    std::string sourceFilename, binaryFilename;
    occa::json properties;
    hash_t hash;

    // Requirements to launch kernel
    dim outerDims, innerDims;
    std::vector<kernelArgData> arguments;
    lang::kernelMetadata_t metadata;

    // References
    gc::ring_t<kernel> kernelRing;

    modeKernel_t(modeDevice_t *modeDevice_,
                 const std::string &name_,
                 const std::string &sourceFilename_,
                 const occa::json &json_);

    void dontUseRefs();
    void addKernelRef(kernel *ker);
    void removeKernelRef(kernel *ker);
    bool needsFree() const;

    void assertArgumentLimit() const;
    void assertArgInDevice(const kernelArgData &arg,
                          const int argIndex) const;

    void setArguments(kernelArg *args,
                      const int count);
    void pushArgument(const kernelArg &arg);

    void setSourceMetadata(lang::parser_t &parser);

    void setupRun();

    bool isNoop() const;

    //---[ Virtual Methods ]------------
    virtual ~modeKernel_t() = 0;

    virtual int maxDims() const = 0;
    virtual dim maxOuterDims() const = 0;
    virtual dim maxInnerDims() const = 0;

    virtual const lang::kernelMetadata_t& getMetadata() const = 0;

    virtual void run() const = 0;
    //==================================
  };
}

#endif
