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
#include <occa/core/device.hpp>
#include <occa/core/memory.hpp>
#include <occa/core/kernelArg.hpp>
#include <occa/tools/uva.hpp>

namespace occa {
  //---[ KernelArg ]--------------------
  kernelArgData::kernelArgData() :
    modeDevice(NULL),
    modeMemory(NULL),
    size(0),
    info(kArgInfo::none) {
    ::memset(&data, 0, sizeof(data));
  }

  kernelArgData::kernelArgData(const kernelArgData &k) {
    *this = k;
  }

  kernelArgData& kernelArgData::operator = (const kernelArgData &k) {
    modeDevice = k.modeDevice;
    modeMemory = k.modeMemory;

    data = k.data;
    size = k.size;
    info = k.info;

    return *this;
  }

  kernelArgData::~kernelArgData() {}

  void* kernelArgData::ptr() const {
    return ((info & kArgInfo::usePointer) ? data.void_ : (void*) &data);
  }

  kernelArg::kernelArg() {}
  kernelArg::~kernelArg() {}

  kernelArg::kernelArg(const kernelArgData &arg) {
    args.push_back(arg);
  }

  kernelArg::kernelArg(const kernelArg &k) :
    args(k.args) {}

  int kernelArg::size() {
    return (int) args.size();
  }

  kernelArgData& kernelArg::operator [] (const int index) {
    return args[index];
  }

  kernelArg& kernelArg::operator = (const kernelArg &k) {
    args = k.args;
    return *this;
  }

  kernelArg::kernelArg(const uint8_t arg) {
    kernelArgData kArg;
    kArg.data.uint8_ = arg;
    kArg.size        = sizeof(uint8_t);
    args.push_back(kArg);
  }

  kernelArg::kernelArg(const uint16_t arg) {
    kernelArgData kArg;
    kArg.data.uint16_ = arg;
    kArg.size         = sizeof(uint16_t);
    args.push_back(kArg);
  }

  kernelArg::kernelArg(const uint32_t arg) {
    kernelArgData kArg;
    kArg.data.uint32_ = arg;
    kArg.size         = sizeof(uint32_t);
    args.push_back(kArg);
  }

  kernelArg::kernelArg(const uint64_t arg) {
    kernelArgData kArg;
    kArg.data.uint64_ = arg;
    kArg.size         = sizeof(uint64_t);
    args.push_back(kArg);
  }

  kernelArg::kernelArg(const int8_t arg) {
    kernelArgData kArg;
    kArg.data.int8_ = arg;
    kArg.size       = sizeof(int8_t);
    args.push_back(kArg);
  }

  kernelArg::kernelArg(const int16_t arg) {
    kernelArgData kArg;
    kArg.data.int16_ = arg;
    kArg.size        = sizeof(int16_t);
    args.push_back(kArg);
  }

  kernelArg::kernelArg(const int32_t arg) {
    kernelArgData kArg;
    kArg.data.int32_ = arg;
    kArg.size        = sizeof(int32_t);
    args.push_back(kArg);
  }

  kernelArg::kernelArg(const int64_t arg) {
    kernelArgData kArg;
    kArg.data.int64_ = arg;
    kArg.size        = sizeof(int64_t);
    args.push_back(kArg);
  }

  kernelArg::kernelArg(const float arg) {
    kernelArgData kArg;
    kArg.data.float_ = arg;
    kArg.size         = sizeof(float);
    args.push_back(kArg);
  }

  kernelArg::kernelArg(const double arg) {
    kernelArgData kArg;
    kArg.data.double_ = arg;
    kArg.size         = sizeof(double);
    args.push_back(kArg);
  }

  void kernelArg::add(const kernelArg &arg) {
    const int newArgs = (int) arg.args.size();
    for (int i = 0; i < newArgs; ++i) {
      args.push_back(arg.args[i]);
    }
  }

  void kernelArg::add(void *arg,
                      bool lookAtUva, bool argIsUva) {
    add(arg, sizeof(void*), lookAtUva, argIsUva);
  }

  void kernelArg::add(void *arg, size_t bytes,
                      bool lookAtUva, bool argIsUva) {

    modeMemory_t *modeMemory = NULL;

    if (argIsUva) {
      modeMemory = (modeMemory_t*) arg;
    } else if (lookAtUva) {
      ptrRangeMap::iterator it = uvaMap.find(arg);
      if (it != uvaMap.end()) {
        modeMemory = it->second;
      }
    }

    if (modeMemory) {
      add(modeMemory->makeKernelArg());
    } else {
      kernelArgData kArg;
      kArg.data.void_ = arg;
      kArg.size       = bytes;
      kArg.info       = kArgInfo::usePointer;
      args.push_back(kArg);
    }
  }

  void kernelArg::setupForKernelCall(const bool isConst) const {
    const int argCount = (int) args.size();
    for (int i = 0; i < argCount; ++i) {
      occa::modeMemory_t *modeMemory = args[i].modeMemory;

      if (!modeMemory              ||
          !modeMemory->isManaged() ||
          !modeMemory->modeDevice->hasSeparateMemorySpace()) {
        continue;
      }
      if (!modeMemory->inDevice()) {
        modeMemory->copyFrom(modeMemory->uvaPtr, modeMemory->size);
        modeMemory->memInfo |= uvaFlag::inDevice;
      }
      if (!isConst && !modeMemory->isStale()) {
        uvaStaleMemory.push_back(modeMemory);
        modeMemory->memInfo |= uvaFlag::isStale;
      }
    }
  }

  int kernelArg::argumentCount(const std::vector<kernelArg> &arguments) {
    const int kArgCount = (int) arguments.size();
    int argc = 0;
    for (int i = 0; i < kArgCount; ++i) {
      argc += arguments[i].args.size();
    }
    return argc;
  }
  //====================================
}
