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

#ifndef OCCA_CORE_KERNELARG_HEADER
#define OCCA_CORE_KERNELARG_HEADER

#include <vector>

#include <occa/defines.hpp>
#include <occa/types.hpp>

namespace occa {
  class modeMemory_t; class memory;
  class modeDevice_t; class device;
  class kernelArgData;

  typedef std::vector<kernelArgData>          kArgVector;
  typedef kArgVector::iterator                kArgVectorIterator;
  typedef kArgVector::const_iterator          cKArgVectorIterator;

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
    occa::modeDevice_t *modeDevice;
    occa::modeMemory_t *modeMemory;

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
}

#endif
