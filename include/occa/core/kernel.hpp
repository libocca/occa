#ifndef OCCA_CORE_KERNEL_HEADER
#define OCCA_CORE_KERNEL_HEADER

#include <initializer_list>
#include <iostream>
#include <stdint.h>
#include <vector>

#include <occa/defines.hpp>
#include <occa/core/kernelArg.hpp>
#include <occa/types.hpp>

// Unfortunately we need to expose this in include
#include <occa/utils/gc.hpp>

namespace occa {
  class modeKernel_t; class kernel;
  class modeMemory_t; class memory;
  class modeDevice_t; class device;
  class kernelBuilder;

  namespace lang {
    class parser_t;
  }

  typedef std::map<hash_t, kernel>            hashedKernelMap;
  typedef hashedKernelMap::iterator           hashedKernelMapIterator;
  typedef hashedKernelMap::const_iterator     cHashedKernelMapIterator;

  typedef std::vector<kernelBuilder>          kernelBuilderVector;
  typedef kernelBuilderVector::iterator       kernelBuilderVectorIterator;
  typedef kernelBuilderVector::const_iterator cKernelBuilderVectorIterator;

  //---[ kernel ]-----------------------
  class kernel : public gc::ringEntry_t {
    friend class occa::modeKernel_t;
    friend class occa::device;

  private:
    modeKernel_t *modeKernel;

  public:
    kernel();
    kernel(modeKernel_t *modeKernel_);

    kernel(const kernel &k);
    kernel& operator = (const kernel &k);
    kernel& operator = (modeKernel_t *modeKernel_);
    ~kernel();

  private:
    void assertInitialized() const;
    void setModeKernel(modeKernel_t *modeKernel_);
    void removeKernelRef();

  public:
    void dontUseRefs();

    bool isInitialized();

    const std::string& mode() const;
    const occa::json& properties() const;

    modeKernel_t* getModeKernel() const;

    occa::device getDevice();

    bool operator == (const occa::kernel &other) const;
    bool operator != (const occa::kernel &other) const;

    const std::string& name();
    const std::string& sourceFilename();
    const std::string& binaryFilename();
    hash_t hash();

    int maxDims();
    dim maxOuterDims();
    dim maxInnerDims();

    void setRunDims(dim outerDims, dim innerDims);

    void pushArg(const kernelArg &arg);
    void clearArgs();

    void run() const;
    void run(std::initializer_list<kernelArg> args) const;

#include "kernelOperators.hpp_codegen"

    void free();
  };
  //====================================


  //---[ Kernel Properties ]------------
  // Properties:
  //   defines       : Object
  //   includes      : Array
  //   header        : Array
  //   include_paths : Array
  hash_t kernelHeaderHash(const occa::json &props);

  std::string assembleKernelHeader(const occa::json &props);
  //====================================
}

#endif
