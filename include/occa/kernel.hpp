/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2017 David Medina and Tim Warburton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */

#ifndef OCCA_KERNEL_HEADER
#define OCCA_KERNEL_HEADER

#include <iostream>
#include <stdint.h>

#include "occa/defines.hpp"
#include "occa/tools/gc.hpp"
#include "occa/tools/properties.hpp"
#include "occa/parser/types.hpp"

namespace occa {
  class kernel_v; class kernel;
  class memory_v; class memory;
  class device_v; class device;
  class kernelArg_t;
  class kernelBuilder;

  typedef std::vector<kernelArg_t>     kArgVector_t;
  typedef kArgVector_t::iterator       kArgVectorIterator;
  typedef kArgVector_t::const_iterator cKArgVectorIterator;

  typedef std::map<hash_t,occa::kernel>     hashedKernelMap_t;
  typedef hashedKernelMap_t::iterator       hashedKernelMapIterator;
  typedef hashedKernelMap_t::const_iterator cHashedKernelMapIterator;

  typedef std::vector<kernelBuilder>            kernelBuilderVector_t;
  typedef kernelBuilderVector_t::iterator       kernelBuilderVectorIterator;
  typedef kernelBuilderVector_t::const_iterator cKernelBuilderVectorIterator;

  //---[ KernelArg ]--------------------
  namespace kArgInfo {
    static const char none       = 0;
    static const char usePointer = (1 << 0);
    static const char hasTexture = (1 << 1);
  }

  union kernelArgData_t {
    uint8_t  uint8_;
    uint16_t uint16_;
    uint32_t uint32_;
    uint64_t uint64_;

    int8_t  int8_;
    int16_t int16_;
    int32_t int32_;
    int64_t int64_;

    float float_;
    double double_;

    void* void_;
  };

  class kernelArg_t {
  public:
    occa::device_v *dHandle;
    occa::memory_v *mHandle;

    kernelArgData_t data;
    udim_t size;
    char info;

    kernelArg_t();
    kernelArg_t(const kernelArg_t &k);
    kernelArg_t& operator = (const kernelArg_t &k);
    ~kernelArg_t();

    void* ptr() const;
  };

  class kernelArg {
  public:
    kArgVector_t args;

    kernelArg();
    ~kernelArg();
    kernelArg(kernelArg_t &arg);
    kernelArg(const kernelArg &k);
    kernelArg& operator = (const kernelArg &k);

    kernelArg(const uint8_t arg);
    kernelArg(const uint16_t arg);
    kernelArg(const uint32_t arg);
    kernelArg(const uint64_t arg);

    kernelArg(const int8_t arg);
    kernelArg(const int16_t arg);
    kernelArg(const int32_t arg);
    kernelArg(const int64_t arg);

    kernelArg(const float arg);
    kernelArg(const double arg);

    template <class TM>
    kernelArg(const type2<TM> &arg) {
      addArg((void*) const_cast<type2<TM>*>(&arg), sizeof(type2<TM>), false);
    }

    template <class TM>
    kernelArg(const type4<TM> &arg) {
      addArg((void*) const_cast<type4<TM>*>(&arg), sizeof(type4<TM>), false);
    }

    template <class TM>
    kernelArg(TM *arg) {
      addArg((void*) arg, true, false);
    }

    template <class TM>
    kernelArg(const TM *arg) {
      addArg((void*) const_cast<TM*>(arg), true, false);
    }

    template <class TM>
    void addArg(const TM &arg) {
      addArg((void*) const_cast<TM*>(&arg), sizeof(TM), false);
    }

    void addArg(void *arg,
                bool lookAtUva = true, bool argIsUva = false);

    void addArg(void *arg, size_t bytes,
                bool lookAtUva = true, bool argIsUva = false);

    void setupForKernelCall(const bool isConst) const;

    static int argumentCount(const int argc, const kernelArg *args);
  };

  template <>
  void kernelArg::addArg(const kernelArg &arg);
  //====================================


  //---[ Kernel Properties ]------------
  std::string assembleHeader(const occa::properties &props);
  //====================================


  //---[ kernel_v ]---------------------
  class kernel_v : public withRefs {
  public:
    occa::device_v *dHandle;

    std::string name;
    std::string sourceFilename, binaryFilename;
    occa::properties properties;

    dim inner, outer;

    std::vector<kernel> nestedKernels;
    std::vector<kernelArg> arguments;

    kernelMetadata metadata;

    kernel_v(const occa::properties &properties_);

    void initFrom(const kernel_v &m);

    kernel* nestedKernelsPtr();
    int nestedKernelCount();

    kernelArg* argumentsPtr();
    int argumentCount();

    std::string binaryName(const std::string &filename);
    std::string getSourceFilename(const std::string &filename, hash_t &hash);
    std::string getBinaryFilename(const std::string &filename, hash_t &hash);

    //---[ Virtual Methods ]------------
    virtual ~kernel_v() = 0;
    // Must be able to be called multiple times safely
    virtual void free() = 0;

    virtual void* getHandle(const occa::properties &props) const = 0;

    virtual void build(const std::string &filename,
                       const std::string &kernelName,
                       const occa::properties &props) = 0;

    virtual void buildFromBinary(const std::string &filename,
                                 const std::string &kernelName,
                                 const occa::properties &props) = 0;

    virtual int maxDims() const = 0;
    virtual dim maxOuterDims() const = 0;
    virtual dim maxInnerDims() const = 0;

    virtual void runFromArguments(const int kArgc, const kernelArg *kArgs) const = 0;
    //==================================
  };
  //====================================

  //---[ kernel ]-----------------------
  class kernel {
    friend class occa::device;

  private:
    kernel_v *kHandle;

  public:
    kernel();
    kernel(kernel_v *kHandle_);

    kernel(const kernel &k);
    kernel& operator = (const kernel &k);
    ~kernel();

  private:
    void setKHandle(kernel_v *kHandle_);
    void setDHandle(device_v *dHandle_);
    void removeKHandleRef();

  public:
    void dontUseRefs();

    bool isInitialized();

    void* getHandle(const occa::properties &props = occa::properties());
    kernel_v* getKHandle();

    occa::device getDevice();

    const std::string& mode();
    const std::string& name();
    const std::string& sourceFilename();
    const std::string& binaryFilename();

    int maxDims();
    dim maxOuterDims();
    dim maxInnerDims();

    void setRunDims(dim outer, dim inner);

    void addArgument(const int argPos, const kernelArg &arg);
    void runFromArguments() const;
    void clearArgumentList();

#include "occa/operators/declarations.hpp"

    void free();
  };
  //====================================

  //---[ kernel builder ]---------------
  class kernelBuilder {
  protected:
    std::string source_;
    std::string function_;
    occa::properties props_;

    hashedKernelMap_t kernelMap;

    bool buildingFromFile;

  public:
    kernelBuilder();

    kernelBuilder(const kernelBuilder &k);
    kernelBuilder& operator = (const kernelBuilder &k);

    static kernelBuilder fromFile(const std::string &filename,
                                  const std::string &function,
                                  const occa::properties &props = occa::properties());

    static kernelBuilder fromString(const std::string &content,
                                    const std::string &function,
                                    const occa::properties &props = occa::properties());

    bool isInitialized();

    virtual occa::kernel build(occa::device device,
                               const hash_t &hash);

    occa::kernel build(occa::device device);

    occa::kernel build(occa::device device,
                       const occa::properties &props);

    occa::kernel build(const int id,
                       occa::device device,
                       const occa::properties &props = occa::properties());

    occa::kernel operator [] (occa::device device);

    void free();
  };
  //====================================
}

#endif
