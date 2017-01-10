/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2016 David Medina and Tim Warburton
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
#include "occa/tools/properties.hpp"
#include "occa/parser/types.hpp"

namespace occa {
  class kernel_v; class kernel;
  class memory_v; class memory;
  class device_v; class device;

  static const bool useParser = true;

  class kernelArg_t;

  typedef std::vector<kernelArg_t>     kArgVector_t;
  typedef kArgVector_t::iterator       kArgVectorIterator;
  typedef kArgVector_t::const_iterator cKArgVectorIterator;

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
    udim_t          size;
    char            info;

    kernelArg_t();
    kernelArg_t(const kernelArg_t &k);
    kernelArg_t& operator = (const kernelArg_t &k);
    ~kernelArg_t();

    void* ptr() const;
  };

  class kernelArg {
  public:
    kernelArg_t arg;
    kArgVector_t extraArgs;

    kernelArg();
    ~kernelArg();
    kernelArg(kernelArg_t &arg_);
    kernelArg(const kernelArg &k);
    kernelArg& operator = (const kernelArg &k);

    template <class TM>
    kernelArg(const TM &arg_) {
      setupFrom(const_cast<TM*>(&arg_), sizeof(TM), false);
    }

    template <class TM>
    kernelArg(TM *arg_) {
      setupFrom(arg_);
    }

    template <class TM>
    kernelArg(const TM *arg_) {
      setupFrom(const_cast<TM*>(arg_));
    }

    void setupFrom(void *arg_,
                   bool lookAtUva = true, bool argIsUva = false);

    void setupFrom(void *arg_, size_t bytes,
                   bool lookAtUva = true, bool argIsUva = false);

    occa::device getDevice() const;

    void setupForKernelCall(const bool isConst) const;

    static int argumentCount(const int argc, const kernelArg *args);
  };

  template <> kernelArg::kernelArg(const uint8_t &arg_);
  template <> kernelArg::kernelArg(const uint16_t &arg_);
  template <> kernelArg::kernelArg(const uint32_t &arg_);
  template <> kernelArg::kernelArg(const uint64_t &arg_);

  template <> kernelArg::kernelArg(const int8_t &arg_);
  template <> kernelArg::kernelArg(const int16_t &arg_);
  template <> kernelArg::kernelArg(const int32_t &arg_);
  template <> kernelArg::kernelArg(const int64_t &arg_);

  template <> kernelArg::kernelArg(const float &arg_);
  template <> kernelArg::kernelArg(const double &arg_);
  //====================================


  //---[ kernelInfo ]---------------------
  class kernelInfo : public occa::properties {
  public:
    kernelInfo();
    kernelInfo(const properties &props);

    static bool isAnOccaDefine(const std::string &name);
    void addIncludeDefine(const std::string &filename);
    void addInclude(const std::string &filename);
    void removeDefine(const std::string &macro);

    template <class TM>
    inline void addDefine(const std::string &macro, const TM &definedValue) {
      std::string &headers = (*this)["headers"].getString();
      if (isAnOccaDefine(macro)) {
        headers += "\n#undef " + macro;
      }
      headers += "\n#define " + macro + " " + occa::toString(definedValue);
    }

    void addSource(const std::string &content);
  };
  //====================================


  //---[ kernel_v ]---------------------
  class kernel_v {
  public:
    occa::device_v *dHandle;

    std::string name;
    std::string sourceFilename, binaryFilename;
    occa::properties properties;

    int dims;
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
    virtual void free() = 0;

    virtual void* getHandle(const occa::properties &props) = 0;

    virtual void build(const std::string &filename,
                       const std::string &functionName,
                       const occa::properties &props) = 0;

    virtual void buildFromBinary(const std::string &filename,
                                 const std::string &functionName,
                                 const occa::properties &props) = 0;

    virtual int maxDims() = 0;
    virtual dim maxOuterDims() = 0;
    virtual dim maxInnerDims() = 0;

    virtual void runFromArguments(const int kArgc, const kernelArg *kArgs) = 0;
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

    void* getHandle(const occa::properties &props = occa::properties());
    kernel_v* getKHandle();

    const std::string& mode();
    const std::string& name();
    const std::string& sourceFilename();
    const std::string& binaryFilename();

    occa::device getDevice();

    int maxDims();
    dim maxOuterDims();
    dim maxInnerDims();

    void setWorkingDims(int dims, dim inner, dim outer);

    void addArgument(const int argPos, const kernelArg &arg);
    void runFromArguments();
    void clearArgumentList();

#include "occa/operators/declarations.hpp"

    void free();
  };
  //====================================
}

#endif
