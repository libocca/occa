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

#include "occa/base.hpp"
#include "occa/tools/sys.hpp"

#define OCCA_C_EXPORTS

#include "occa/c_wrapper.hpp"

namespace occa {
  namespace c {
    enum type {
      none_,

      ptr_,
      int8_ , uint8_,
      int16_, uint16_,
      int32_, uint32_,
      int64_, uint64_,
      float_, double_,

      struct_,
      string_,

      device_,
      kernel_,
      memory_,
      stream_,

      properties_,

      argumentList_
    };
  }

  template <class TM>
  class ctype {
  public:
    static const c::type type;
  };

  template <>
  const c::type ctype<int8_t>::type = c::int8_;
  template <>
  const c::type ctype<uint8_t>::type = c::uint8_;

  template <>
  const c::type ctype<int16_t>::type = c::int16_;
  template <>
  const c::type ctype<uint16_t>::type = c::uint16_;

  template <>
  const c::type ctype<int32_t>::type = c::int32_;
  template <>
  const c::type ctype<uint32_t>::type = c::uint32_;

  template <>
  const c::type ctype<int64_t>::type = c::int64_;
  template <>
  const c::type ctype<uint64_t>::type = c::uint64_;

  template <>
  const c::type ctype<float>::type = c::float_;
  template <>
  const c::type ctype<double>::type = c::double_;
}

OCCA_START_EXTERN_C

struct occaObject_t {
  occa::c::type type;
  void *obj;

  inline occaObject_t() :
    type(),
    obj() {}

  inline occaObject_t(occa::c::type type_) :
    type(type_),
    obj() {}
};

OCCA_END_EXTERN_C

namespace occa {
  namespace c {
    inline occaObject newObject(void *obj, const occa::c::type type_ = none_) {
      occaObject o;
      o.ptr = new occaObject_t;
      o.ptr->obj  = obj;
      o.ptr->type = type_;
      return o;
    }

    inline occaObject newObject(occaObject_t &oo) {
      occaObject o;
      o.ptr = &oo;
      return o;
    }
  }
}

OCCA_START_EXTERN_C

struct occaArgumentList_t {
  int argc;
  occaObject argv[100];
};

//---[ Globals & Flags ]----------------
const occaObject nullObject = occa::c::newObject(NULL);
const occaObject occaDefault = nullObject;
const occaUDim_t occaAllBytes = -1;
const void *occaEmptyProperties = (void*) new occa::properties();

void OCCA_RFUNC occaSetVerboseCompilation(const int value) {
  occa::settings()["verboseCompilation"] = (bool) value;
}
//======================================

OCCA_END_EXTERN_C

//---[ Types ]--------------------------
namespace occa {
  namespace c {
    inline occaType newType(const occa::c::type type) {
      occaType ot;
      ot.ptr = new occaObject_t();
      ot.ptr->type = type;
      ot.ptr->obj = new occa::kernelArgData();
      return ot;
    }

    template <class TM>
    inline void free(const occaObject &t) {
      delete (TM*) t.ptr->obj;
      delete t.ptr;
    }

    inline void freePtr(const occaType &t) {
      delete t.ptr;
    }

    template <>
    inline void free<void>(const occaType &t) {
      switch (t.ptr->type) {
      case int8_  : occa::c::free<int8_t>(t); break;
      case uint8_ : occa::c::free<uint8_t>(t); break;
      case int16_ : occa::c::free<int16_t>(t); break;
      case uint16_: occa::c::free<uint16_t>(t); break;
      case int32_ : occa::c::free<int32_t>(t); break;
      case uint32_: occa::c::free<uint32_t>(t); break;
      case int64_ : occa::c::free<int64_t>(t); break;
      case uint64_: occa::c::free<uint64_t>(t); break;

      case float_  : occa::c::free<float>(t); break;
      case double_ : occa::c::free<double>(t); break;

      case string_ : occa::c::freePtr(t); break;
      case struct_ : occa::c::freePtr(t); break;
      default: break;
      }
    }

    template <class TM>
    inline TM& from(const occaObject &t) {
      return *((TM*) t.ptr->obj);
    }

    template <class TM>
    inline TM* fromPtr(const occaType &t) {
      return (TM*) t.ptr->obj;
    }

    inline bool sameObject(const occaObject &a, const occaObject &b) {
      return (a.ptr->obj == b.ptr->obj);
    }

    inline bool isDefault(const occaObject &obj) {
      return sameObject(obj, occaDefault);
    }

    inline type& getType(occaType &t) {
      return t.ptr->type;
    }

    inline occa::kernelArgData& getKernelArg(const occaObject &t) {
      return occa::c::from<occa::kernelArgData>(t);
    }

    inline occa::device getDevice(const occaDevice &device) {
      return occa::device(occa::c::fromPtr<occa::device_v>(device));
    }

    inline occa::kernel getKernel(const occaKernel &kernel) {
      return occa::kernel(occa::c::fromPtr<occa::kernel_v>(kernel));
    }

    inline occa::memory getMemory(const occaMemory &memory) {
      return occa::memory((memory_v*) getKernelArg(memory).data.void_);
    }

    inline occa::properties &getProperties(const occaProperties &properties) {
      if (isDefault(properties)) {
        return *((occa::properties*) occaEmptyProperties);
      }
      return occa::c::getProperties(properties);
    }

    inline occaType createOccaType(void *ptr, size_t bytes, occa::c::type type) {
      occaType ot = occa::c::newType(type);
      occa::kernelArgData &kArg = occa::c::getKernelArg(ot);
      if ((type == ptr_) || (type == struct_) || (type == string_)) {
        kArg.info = occa::kArgInfo::usePointer;
      }
      kArg.size = bytes;
      memcpy(&kArg.data, ptr, bytes);
      return ot;
    }

    template <class TM>
    inline occaType createOccaType(TM *ptr) {
      const occa::c::type type = occa::ctype<TM>::type;
      const size_t bytes = sizeof(TM);
      return createOccaType(ptr, bytes, type);
    }

    inline std::string toString(occaType value) {
      occa::kernelArgData &value_ = occa::c::getKernelArg(value);
      const int valueType         = occa::c::getType(value);

      switch (valueType) {
      case int8_  : return occa::toString(value_.data.int8_);
      case uint8_ : return occa::toString(value_.data.uint8_);
      case int16_ : return occa::toString(value_.data.int16_);
      case uint16_: return occa::toString(value_.data.uint16_);
      case int32_ : return occa::toString(value_.data.int32_);
      case uint32_: return occa::toString(value_.data.uint32_);
      case int64_ : return occa::toString(value_.data.int64_);
      case uint64_: return occa::toString(value_.data.uint64_);

      case float_  : return occa::toString(value_.data.float_);
      case double_ : return occa::toString(value_.data.double_);

      case string_ : return std::string((char*) value_.data.void_);
      default:
        std::cout << "Wrong type input in [occa::c::typeToStr]\n";
      }

      return "";
    }
  }
}

OCCA_START_EXTERN_C

//  ---[ Known Types ]------------------
OCCA_LFUNC occaType OCCA_RFUNC occaPtr(void *value) {
  return occa::c::createOccaType(&value, sizeof(value), occa::c::ptr_);
}

OCCA_LFUNC occaType OCCA_RFUNC occaInt8(int8_t value) {
  return occa::c::createOccaType(&value);
}

OCCA_LFUNC occaType OCCA_RFUNC occaUInt8(uint8_t value) {
  return occa::c::createOccaType(&value);
}

OCCA_LFUNC occaType OCCA_RFUNC occaInt16(int16_t value) {
  return occa::c::createOccaType(&value);
}

OCCA_LFUNC occaType OCCA_RFUNC occaUInt16(uint16_t value) {
  return occa::c::createOccaType(&value);
}

OCCA_LFUNC occaType OCCA_RFUNC occaInt32(int32_t value) {
  return occa::c::createOccaType(&value);
}

OCCA_LFUNC occaType OCCA_RFUNC occaUInt32(uint32_t value) {
  return occa::c::createOccaType(&value);
}

OCCA_LFUNC occaType OCCA_RFUNC occaInt64(int64_t value) {
  return occa::c::createOccaType(&value);
}

OCCA_LFUNC occaType OCCA_RFUNC occaUInt64(uint64_t value) {
  return occa::c::createOccaType(&value);
}
//  ====================================

//  ---[ Ambiguous Types ]--------------
occaType OCCA_RFUNC occaChar(char value) {
  return occa::c::createOccaType(&value, sizeof(value), occa::c::int8_);
}

occaType OCCA_RFUNC occaUChar(unsigned char value) {
  return occa::c::createOccaType(&value, sizeof(value), occa::c::uint8_);
}

occaType OCCA_RFUNC occaShort(short value) {
  return occa::c::createOccaType(&value, sizeof(value), occa::c::int16_);
}

occaType OCCA_RFUNC occaUShort(unsigned short value) {
  return occa::c::createOccaType(&value, sizeof(value), occa::c::uint16_);
}

occaType OCCA_RFUNC occaInt(int value) {
  return occa::c::createOccaType(&value, sizeof(value), occa::c::int32_);
}

occaType OCCA_RFUNC occaUInt(unsigned int value) {
  return occa::c::createOccaType(&value, sizeof(value), occa::c::uint32_);
}

occaType OCCA_RFUNC occaLong(long value) {
  return occa::c::createOccaType(&value, sizeof(value), occa::c::int64_);
}

occaType OCCA_RFUNC occaULong(unsigned long value) {
  return occa::c::createOccaType(&value, sizeof(value), occa::c::uint64_);
}

occaType OCCA_RFUNC occaFloat(float value) {
  return occa::c::createOccaType(&value);
}

occaType OCCA_RFUNC occaDouble(double value) {
  return occa::c::createOccaType(&value);
}

occaType OCCA_RFUNC occaStruct(void *value, occaUDim_t bytes) {
  return occa::c::createOccaType(&value, bytes, occa::c::struct_);
}

occaType OCCA_RFUNC occaString(const char *value) {
  return occa::c::createOccaType(&value,
                                 sizeof(value),
                                 occa::c::string_);
}
//  ====================================

//  ---[ Properties ]-------------------
occaObject OCCA_RFUNC occaCreateProperties() {
  occa::properties &newProperties = *(new occa::properties());
  return newObject(&newProperties, occa::c::properties_);
}

occaObject OCCA_RFUNC occaCreatePropertiesFromString(const char *c) {
  occa::properties &newProperties = *(new occa::properties(c));
  return newObject(&newProperties, occa::c::properties_);
}

void OCCA_RFUNC occaPropertiesSet(occaProperties properties,
                                  const char *key,
                                  occaType value) {
  occa::properties &props_    = occa::c::from<occa::properties>(properties);
  occa::kernelArgData &value_ = occa::c::getKernelArg(value);
  const int valueType         = occa::c::getType(value);

  switch (valueType) {
  case occa::c::int8_  : {
    props_[key] = value_.data.int8_;
    break;
  }
  case occa::c::uint8_ : {
    props_[key] = value_.data.uint8_;
    break;
  }
  case occa::c::int16_ : {
    props_[key] = value_.data.int16_;
    break;
  }
  case occa::c::uint16_: {
    props_[key] = value_.data.uint16_;
    break;
  }
  case occa::c::int32_ : {
    props_[key] = value_.data.int32_;
    break;
  }
  case occa::c::uint32_: {
    props_[key] = value_.data.uint32_;
    break;
  }
  case occa::c::int64_ : {
    props_[key] = value_.data.int64_;
    break;
  }
  case occa::c::uint64_: {
    props_[key] = value_.data.uint64_;
    break;
  }
  case occa::c::float_  : {
    props_[key] = value_.data.float_;
    break;
  }
  case occa::c::double_ : {
    props_[key] = value_.data.double_;
    break;
  }
  case occa::c::string_ : {
    props_[key] = (char*) value_.data.void_;
    break;
  }
  default: {
    std::cout << "Wrong type input in [occaPropertiesSet]\n";
  }}
}

void OCCA_RFUNC occaPropertiesFree(occaProperties properties) {
  occa::c::free<occa::properties>(properties);
}
//  ====================================
//======================================


//---[ Background Device ]--------------
//  |---[ Device ]----------------------
void OCCA_RFUNC occaSetDevice(occaDevice device) {
  occa::device device_ = occa::c::getDevice(device);
  device_.dontUseRefs();
  occa::setDevice(device_);
}

void OCCA_RFUNC occaSetDeviceFromInfo(const char *infos) {
  occa::setDevice(infos);
}

occaDevice OCCA_RFUNC occaGetDevice() {
  occa::device device = occa::getDevice();
  return newObject(device.getDHandle(), occa::c::device_);
}

void OCCA_RFUNC occaFinish() {
  occa::finish();
}

void OCCA_RFUNC occaWaitFor(occaStreamTag tag) {
  occa::streamTag tag_;
  ::memcpy(&tag_, &tag, sizeof(tag_));
  occa::waitFor(tag_);
}

occaStream OCCA_RFUNC occaCreateStream() {
  occa::stream &newStream = *(new occa::stream(occa::createStream()));
  return newObject(&newStream, occa::c::stream_);
}

occaStream OCCA_RFUNC occaGetStream() {
  occa::stream &currentStream = *(new occa::stream(occa::getStream()));
  return newObject(&currentStream, occa::c::stream_);
}

void OCCA_RFUNC occaSetStream(occaStream stream) {
  occa::setStream(occa::c::from<occa::stream>(stream));
}

occaStream OCCA_RFUNC occaWrapStream(void *handle_, const occaProperties props) {
  occa::properties &props_ = occa::c::from<occa::properties>(props);
  occa::stream &newStream = *(new occa::stream(occa::wrapStream(handle_, props_)));
  return newObject(&newStream, occa::c::stream_);
}

occaStreamTag OCCA_RFUNC occaTagStream() {
  occa::streamTag oldTag = occa::tagStream();
  occaStreamTag newTag;

  newTag.tagTime = oldTag.tagTime;
  ::memcpy(&(newTag.handle), &(oldTag.handle), sizeof(void*));

  return newTag;
}
//  |===================================

//  |---[ Kernel ]----------------------
occaKernel OCCA_RFUNC occaBuildKernel(const char *filename,
                                      const char *kernelName,
                                      const occaProperties props) {
  occa::kernel kernel;

  if (occa::c::isDefault(props)) {
    kernel = occa::buildKernel(filename, kernelName);
  } else {
    occa::properties &props_ = occa::c::from<occa::properties>(props);
    kernel = occa::buildKernel(filename, kernelName, props_);
  }
  kernel.dontUseRefs();

  return newObject(kernel.getKHandle(), occa::c::kernel_);
}

occaKernel OCCA_RFUNC occaBuildKernelFromString(const char *str,
                                                const char *kernelName,
                                                const occaProperties props) {
  occa::kernel kernel;

  if (occa::c::isDefault(props)) {
    kernel = occa::buildKernelFromString(str, kernelName);
  } else {
    occa::properties &props_ = occa::c::from<occa::properties>(props);
    kernel = occa::buildKernelFromString(str, kernelName, props_);
  }
  kernel.dontUseRefs();

  return newObject(kernel.getKHandle(), occa::c::kernel_);
}

occaKernel OCCA_RFUNC occaBuildKernelFromBinary(const char *filename,
                                                const char *kernelName) {

  occa::kernel kernel = occa::buildKernelFromBinary(filename, kernelName);
  kernel.dontUseRefs();
  return newObject(kernel.getKHandle(), occa::c::kernel_);
}
//  |===================================

//  |---[ Memory ]----------------------
void OCCA_RFUNC occaMemorySwap(occaMemory a, occaMemory b) {
  occaObject_t *a_ptr = a.ptr;
  a.ptr = b.ptr;
  b.ptr = a_ptr;
}

occaMemory OCCA_RFUNC occaMalloc(const occaUDim_t bytes,
                                 const void *src,
                                 occaProperties props) {
  occa::properties &props_ = occa::c::getProperties(props);
  occa::memory memory_ = occa::malloc(bytes, src, props_);
  memory_.dontUseRefs();

  occaMemory memory = occa::c::newType(occa::c::memory_);
  occa::c::getKernelArg(memory).data.void_ = memory_.getMHandle();
  return memory;
}

void* OCCA_RFUNC occaUmalloc(const occaUDim_t bytes,
                             const void *src,
                             occaProperties props) {

  occa::properties &props_ = occa::c::getProperties(props);
  void *ptr = occa::umalloc(bytes, src, props_);

  occa::memory memory(ptr);
  memory.dontUseRefs();

  return ptr;
}
//  |===================================
//======================================


//---[ Device ]-------------------------
void OCCA_RFUNC occaPrintModeInfo() {
  occa::printModeInfo();
}

occaDevice OCCA_RFUNC occaCreateDevice(occaObject info) {
  occa::device device;
  if (info.ptr->type == occa::c::string_) {
    device = occa::device((const char*) occa::c::getKernelArg(info).data.void_);
    occa::c::freePtr(info);
  } else if (info.ptr->type == occa::c::properties_) {
    device = occa::device(occa::c::getProperties(info));
  }
  device.dontUseRefs();
  return newObject(device.getDHandle(), occa::c::device_);
}

const char* OCCA_RFUNC occaDeviceMode(occaDevice device) {
  occa::device device_ = occa::c::getDevice(device);
  return device_.mode().c_str();
}

occaUDim_t OCCA_RFUNC occaDeviceMemorySize(occaDevice device) {
  occa::device device_ = occa::c::getDevice(device);
  return device_.memorySize();
}

occaUDim_t OCCA_RFUNC occaDeviceMemoryAllocated(occaDevice device) {
  occa::device device_ = occa::c::getDevice(device);
  return device_.memoryAllocated();
}

occaKernel OCCA_RFUNC occaDeviceBuildKernel(occaDevice device,
                                            const char *filename,
                                            const char *kernelName,
                                            const occaProperties props) {
  occa::device device_ = occa::c::getDevice(device);
  occa::kernel kernel;

  if (occa::c::isDefault(props)) {
    kernel = device_.buildKernel(filename, kernelName);
  } else {
    occa::properties &props_ = occa::c::from<occa::properties>(props);
    kernel = device_.buildKernel(filename, kernelName, props_);
  }
  kernel.dontUseRefs();

  return newObject(kernel.getKHandle(), occa::c::kernel_);
}

occaKernel OCCA_RFUNC occaDeviceBuildKernelFromString(occaDevice device,
                                                      const char *str,
                                                      const char *kernelName,
                                                      const occaProperties props) {
  occa::device device_ = occa::c::getDevice(device);
  occa::kernel kernel;

  if (occa::c::isDefault(props)) {
    kernel = device_.buildKernelFromString(str, kernelName);
  } else {
    occa::properties &props_ = occa::c::from<occa::properties>(props);
    kernel = device_.buildKernelFromString(str, kernelName, props_);
  }
  kernel.dontUseRefs();

  return newObject(kernel.getKHandle(), occa::c::kernel_);
}

occaKernel OCCA_RFUNC occaDeviceBuildKernelFromBinary(occaDevice device,
                                                      const char *filename,
                                                      const char *kernelName) {
  occa::device device_ = occa::c::getDevice(device);
  occa::kernel kernel = device_.buildKernelFromBinary(filename, kernelName);
  kernel.dontUseRefs();
  return newObject(kernel.getKHandle(), occa::c::kernel_);
}

occaMemory OCCA_RFUNC occaDeviceMalloc(occaDevice device,
                                       const occaUDim_t bytes,
                                       const void *src,
                                       occaProperties props) {

  occa::device device_ = occa::c::getDevice(device);
  occa::properties &props_ = occa::c::getProperties(props);
  occa::memory memory_ = device_.malloc(bytes, src, props_);
  memory_.dontUseRefs();

  occaMemory memory = occa::c::newType(occa::c::memory_);
  occa::c::getKernelArg(memory).data.void_ = memory_.getMHandle();
  return memory;
}

void* OCCA_RFUNC occaDeviceUmalloc(occaDevice device,
                                   const occaUDim_t bytes,
                                   const void *src,
                                   occaProperties props) {

  occa::device device_ = occa::c::getDevice(device);
  occa::properties &props_ = occa::c::getProperties(props);
  void *ptr = device_.umalloc(bytes, src, props_);

  occa::memory memory(ptr);
  memory.dontUseRefs();

  return ptr;
}

void OCCA_RFUNC occaDeviceFinish(occaDevice device) {
  occa::device device_ = occa::c::getDevice(device);
  device_.finish();
}

occaStream OCCA_RFUNC occaDeviceCreateStream(occaDevice device) {
  occa::device device_ = occa::c::getDevice(device);
  occa::stream &newStream = *(new occa::stream(device_.createStream()));
  return newObject(&newStream, occa::c::stream_);
}

occaStream OCCA_RFUNC occaDeviceGetStream(occaDevice device) {
  occa::device device_ = occa::c::getDevice(device);
  occa::stream &currentStream = *(new occa::stream(device_.getStream()));
  return newObject(&currentStream, occa::c::stream_);
}

void OCCA_RFUNC occaDeviceSetStream(occaDevice device, occaStream stream) {
  occa::device device_ = occa::c::getDevice(device);
  device_.setStream(*((occa::stream*) stream.ptr->obj));
}

occaStream OCCA_RFUNC occaDeviceWrapStream(occaDevice device,
                                           void *handle_,
                                           const occaProperties props) {
  occa::device device_ = occa::c::getDevice(device);
  occa::properties &props_ = occa::c::from<occa::properties>(props);
  occa::stream &newStream = *(new occa::stream(device_.wrapStream(handle_, props_)));
  return newObject(&newStream, occa::c::stream_);
}

occaStreamTag OCCA_RFUNC occaDeviceTagStream(occaDevice device) {
  occa::device device_ = occa::c::getDevice(device);

  occa::streamTag oldTag = device_.tagStream();
  occaStreamTag newTag;
  ::memcpy(&newTag, &oldTag, sizeof(oldTag));

  return newTag;
}

void OCCA_RFUNC occaDeviceWaitFor(occaDevice device,
                                  occaStreamTag tag) {
  occa::device device_ = occa::c::getDevice(device);

  occa::streamTag tag_;
  ::memcpy(&tag_, &tag, sizeof(tag_));

  device_.waitFor(tag_);
}

double OCCA_RFUNC occaDeviceTimeBetweenTags(occaDevice device,
                                            occaStreamTag startTag, occaStreamTag endTag) {
  occa::device device_ = occa::c::getDevice(device);

  occa::streamTag startTag_, endTag_;
  ::memcpy(&startTag_, &startTag, sizeof(startTag_));
  ::memcpy(&endTag_  , &endTag  , sizeof(endTag_));

  return device_.timeBetween(startTag_, endTag_);
}

void OCCA_RFUNC occaStreamFree(occaStream stream) {
  occa::c::from<occa::stream>(stream).free();
  occa::c::free<occa::stream>(stream);
}

void OCCA_RFUNC occaDeviceFree(occaDevice device) {
  occa::c::getDevice(device).free();
  occa::c::freePtr(device);
}
//======================================


//---[ Kernel ]-------------------------
occaDim OCCA_RFUNC occaCreateDim(occaUDim_t x, occaUDim_t y, occaUDim_t z) {
  occaDim ret;
  ret.x = x;
  ret.y = y;
  ret.z = z;
  return ret;
}

const char* OCCA_RFUNC occaKernelMode(occaKernel kernel) {
  occa::kernel kernel_ = occa::c::getKernel(kernel);
  return kernel_.mode().c_str();
}

const char* OCCA_RFUNC occaKernelName(occaKernel kernel) {
  occa::kernel kernel_ = occa::c::getKernel(kernel);
  return kernel_.name().c_str();
}

occaDevice OCCA_RFUNC occaKernelGetDevice(occaKernel kernel) {
  occa::kernel kernel_ = occa::c::getKernel(kernel);
  occa::device device = kernel_.getDevice();
  return newObject(device.getDHandle(), occa::c::device_);
}

void OCCA_RFUNC occaKernelSetRunDims(occaKernel kernel,
                                     occaDim groups,
                                     occaDim items) {

  occa::kernel kernel_ = occa::c::getKernel(kernel);
  kernel_.setRunDims(occa::dim(groups.x, groups.y, groups.z),
                     occa::dim(items.x, items.y, items.z));
}


occaArgumentList OCCA_RFUNC occaCreateArgumentList() {
  occaArgumentList_t *list = new occaArgumentList_t();
  list->argc = 0;
  return occa::c::newObject(list, occa::c::argumentList_);
}

void OCCA_RFUNC occaArgumentListClear(occaArgumentList list) {
  occaArgumentList_t &list_ = occa::c::from<occaArgumentList_t>(list);
  for (int i = 0; i < list_.argc; ++i) {
    if (list_.argv[i].ptr->type != occa::c::memory_) {
      occa::c::free<void>(list_.argv[i]);
    }
  }
  list_.argc = 0;
}

void OCCA_RFUNC occaArgumentListFree(occaArgumentList list) {
  occa::c::free<occaArgumentList_t>(list);
}

void OCCA_RFUNC occaArgumentListAddArg(occaArgumentList list,
                                       int argPos,
                                       occaObject type) {

  occaArgumentList_t &list_ = occa::c::from<occaArgumentList_t>(list);
  if (list_.argc < (argPos + 1)) {
    OCCA_ERROR("Kernels can only have at most [" << OCCA_MAX_ARGS << "] arguments,"
               << " [" << argPos << "] arguments were set",
               argPos < OCCA_MAX_ARGS);

    list_.argc = (argPos + 1);
  }
  list_.argv[argPos].ptr = type.ptr;
}

// Note the _
// [occaKernelRun] is reserved for a variadic macro which is more user-friendly
void OCCA_RFUNC occaKernelRun_(occaKernel kernel,
                               occaArgumentList list) {

  occaArgumentList_t &list_ = occa::c::from<occaArgumentList_t>(list);
  occaKernelRunN(kernel, list_.argc, list_.argv);
}

void OCCA_RFUNC occaKernelRunN(occaKernel kernel, const int argc, occaType *args) {
  occa::kernel kernel_ = occa::c::getKernel(kernel);
  kernel_.clearArgumentList();

  for (int i = 0; i < argc; ++i) {
    occa::kernelArgData &arg = occa::c::getKernelArg(args[i]);
    void *argPtr = arg.data.void_;

    if (args[i].ptr->type == occa::c::memory_) {
      occa::memory memory_((occa::memory_v*) argPtr);
      kernel_.addArgument(i, memory_);
    } else {
      kernel_.addArgument(i, occa::kernelArg(arg));
      delete args[i].ptr;
    }
  }

  kernel_.runFromArguments();
}

#include "operators/cKernelOperators.cpp"

void OCCA_RFUNC occaKernelFree(occaKernel kernel) {
  occa::c::getKernel(kernel).free();
  occa::c::freePtr(kernel);
}
//======================================


//---[ Memory ]-------------------------
const char* OCCA_RFUNC occaMemoryMode(occaMemory memory) {
  return occa::c::getMemory(memory).mode().c_str();
}

void OCCA_RFUNC occaMemcpy(void *dest, const void *src,
                           const occaUDim_t bytes,
                           occaProperties props) {
  if (!occa::c::isDefault(props)) {
    occa::memcpy(dest, src, bytes);
  } else {
    occa::memcpy(dest, src, bytes, occa::c::getProperties(props));
  }
}

void OCCA_RFUNC occaCopyMemToMem(occaMemory dest, occaMemory src,
                                 const occaUDim_t bytes,
                                 const occaUDim_t destOffset,
                                 const occaUDim_t srcOffset,
                                 occaProperties props) {

  occa::memory src_ = occa::c::getMemory(src);
  occa::memory dest_ = occa::c::getMemory(dest);

  if (!occa::c::isDefault(props)) {
    memcpy(dest_, src_, bytes, destOffset, srcOffset);
  } else {
    memcpy(dest_, src_, bytes, destOffset, srcOffset, occa::c::getProperties(props));
  }
}

void OCCA_RFUNC occaCopyPtrToMem(occaMemory dest, const void *src,
                                 const occaUDim_t bytes,
                                 const occaUDim_t offset,
                                 occaProperties props) {

  occa::memory dest_ = occa::c::getMemory(dest);

  if (!occa::c::isDefault(props)) {
    memcpy(dest_, src, bytes, offset);
  } else {
    memcpy(dest_, src, bytes, offset, occa::c::getProperties(props));
  }
}

void OCCA_RFUNC occaCopyMemToPtr(void *dest, occaMemory src,
                                 const occaUDim_t bytes,
                                 const occaUDim_t offset,
                                 occaProperties props) {

  occa::memory src_ = occa::c::getMemory(src);

  if (!occa::c::isDefault(props)) {
    memcpy(dest, src_, bytes, offset);
  } else {
    memcpy(dest, src_, bytes, offset, occa::c::getProperties(props));
  }
}

void OCCA_RFUNC occaMemoryFree(occaMemory memory) {
  occa::c::getMemory(memory).free();
  occa::c::freePtr(memory);
}
//======================================

//---[ Misc ]---------------------------
void OCCA_RFUNC occaFree(occaObject obj) {
  switch (obj.ptr->type) {
  case occa::c::device_: {
    occaDeviceFree(obj);
    break;
  }
  case occa::c::kernel_: {
    occaKernelFree(obj);
    break;
  }
  case occa::c::memory_: {
    occaMemoryFree(obj);
    break;
  }
  case occa::c::stream_: {
    occaStreamFree(obj);
    break;
  }
  case occa::c::properties_: {
    occaPropertiesFree(obj);
    break;
  }
  case occa::c::argumentList_: {
    occaArgumentListFree(obj);
    break;
  }
  default: {
    occa::c::free<void>(obj);
  }}
}
//======================================

OCCA_END_EXTERN_C
