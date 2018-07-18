/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 David Medina and Tim Warburton
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

#include <occa/defines.hpp>
#include <occa/tools/gc.hpp>
#include <occa/tools/properties.hpp>
#include <occa/lang/kernelMetadata.hpp>

namespace occa {
  class kernel_v; class kernel;
  class memory_v; class memory;
  class device_v; class device;
  class kernelArgData;
  class kernelBuilder;

  namespace lang {
    class parser_t;
  }

  typedef std::vector<kernelArgData>          kArgVector;
  typedef kArgVector::iterator                kArgVectorIterator;
  typedef kArgVector::const_iterator          cKArgVectorIterator;

  typedef std::map<hash_t, kernel>            hashedKernelMap;
  typedef hashedKernelMap::iterator           hashedKernelMapIterator;
  typedef hashedKernelMap::const_iterator     cHashedKernelMapIterator;

  typedef std::vector<kernelBuilder>          kernelBuilderVector;
  typedef kernelBuilderVector::iterator       kernelBuilderVectorIterator;
  typedef kernelBuilderVector::const_iterator cKernelBuilderVectorIterator;

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

  class kernelArgData {
  public:
    occa::device_v *dHandle;
    occa::memory_v *mHandle;

    kernelArgData_t data;
    udim_t size;
    char info;

    kernelArgData();
    kernelArgData(const kernelArgData &k);
    kernelArgData& operator = (const kernelArgData &k);
    ~kernelArgData();

    void* ptr() const;
  };

  class kernelArg {
  public:
    kArgVector args;

    kernelArg();
    ~kernelArg();
    kernelArg(const kernelArgData &arg);
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
      add((void*) const_cast<type2<TM>*>(&arg), sizeof(type2<TM>), false);
    }

    template <class TM>
    kernelArg(const type4<TM> &arg) {
      add((void*) const_cast<type4<TM>*>(&arg), sizeof(type4<TM>), false);
    }

    template <class TM>
    kernelArg(TM *arg) {
      add((void*) arg, true, false);
    }

    template <class TM>
    kernelArg(const TM *arg) {
      add((void*) const_cast<TM*>(arg), true, false);
    }

    int size();

    kernelArgData& operator [] (const int index);

    void add(const kernelArg &arg);

    void add(void *arg,
             bool lookAtUva = true, bool argIsUva = false);

    void add(void *arg, size_t bytes,
             bool lookAtUva = true, bool argIsUva = false);

    void setupForKernelCall(const bool isConst) const;

    static int argumentCount(const std::vector<kernelArg> &arguments);
  };
  //====================================


  //---[ Kernel Properties ]------------
  occa::properties getKernelProperties();

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

    std::vector<kernelArg> arguments;
    lang::kernelMetadata metadata;

    kernel_v(device_v *dHandle_,
             const std::string &name_,
             const std::string &sourceFilename_,
             const occa::properties &properties_);

    kernelArg* argumentsPtr();
    int argumentCount();

    void setMetadata(lang::parser_t &parser);

    //---[ Virtual Methods ]------------
    virtual ~kernel_v() = 0;

    // Must be able to be called multiple times safely
    virtual void free() = 0;

    virtual int maxDims() const = 0;
    virtual dim maxOuterDims() const = 0;
    virtual dim maxInnerDims() const = 0;

    virtual void run() const = 0;
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
    kernel& operator = (kernel_v *kHandle_);
    ~kernel();

  private:
    void setKHandle(kernel_v *kHandle_);
    void removeKHandleRef();

  public:
    void dontUseRefs();

    bool isInitialized();

    const std::string& mode() const;
    const occa::properties& properties() const;

    kernel_v* getKHandle();

    occa::device getDevice();

    const std::string& name();
    const std::string& sourceFilename();
    const std::string& binaryFilename();

    int maxDims();
    dim maxOuterDims();
    dim maxInnerDims();

    void setRunDims(dim outer, dim inner);

    void addArgument(const int argPos, const kernelArg &arg);
    void run() const;
    void clearArgumentList();

#include "kernelOperators.hpp"

    void free();
  };
  //====================================

  //---[ kernelBuilder ]----------------
  class kernelBuilder {
  protected:
    std::string source_;
    std::string function_;
    occa::properties props_;

    hashedKernelMap kernelMap;

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

    occa::kernel build(occa::device device);

    occa::kernel build(occa::device device,
                       const occa::properties &props);

    occa::kernel build(occa::device device,
                       const hash_t &hash);

    occa::kernel build(occa::device device,
                       const hash_t &hash,
                       const occa::properties &props);

    occa::kernel operator [] (occa::device device);

    void free();
  };
  //====================================
}

#endif
