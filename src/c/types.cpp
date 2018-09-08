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

#include <occa/c/types.hpp>

namespace occa {
  namespace c {
    template <class TM>
    inline occaType newOccaIntType(bool isUnsigned,
                                   TM value) {
      switch (sizeof(value)) {
      case 1: return isUnsigned ? occaUInt8(value) : occaInt8(value);
      case 2: return isUnsigned ? occaUInt16(value) : occaInt16(value);
      case 4: return isUnsigned ? occaUInt32(value) : occaInt32(value);
      case 8: return isUnsigned ? occaUInt64(value) : occaInt64(value);
      }
      OCCA_FORCE_ERROR("Unknown int type");
    }

    occaType undefinedOccaType() {
      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_UNDEFINED_HEADER;
      oType.type = occa::c::typeType::undefined;
      oType.value.ptr = NULL;
      oType.needsFree = false;
      return oType;
    }

    occaType defaultOccaType() {
      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type = occa::c::typeType::default_;
      oType.value.ptr = NULL;
      oType.needsFree = false;
      return oType;
    }

    occaType newOccaType(void *value) {
      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type  = typeType::ptr;
      oType.bytes = sizeof(void*);
      oType.value.ptr = (char*) value;
      oType.needsFree = false;
      return oType;
    }

    template <>
    occaType newOccaType(const occa::primitive &value) {
      switch(value.type) {
      case occa::primitiveType::int8_   : return newOccaType<int8_t>(value);
      case occa::primitiveType::uint8_  : return newOccaType<uint8_t>(value);
      case occa::primitiveType::int16_  : return newOccaType<int16_t>(value);
      case occa::primitiveType::uint16_ : return newOccaType<uint16_t>(value);
      case occa::primitiveType::int32_  : return newOccaType<int32_t>(value);
      case occa::primitiveType::uint32_ : return newOccaType<uint32_t>(value);
      case occa::primitiveType::int64_  : return newOccaType<int64_t>(value);
      case occa::primitiveType::uint64_ : return newOccaType<uint64_t>(value);
      case occa::primitiveType::float_  : return newOccaType<float>(value);
      case occa::primitiveType::double_ : return newOccaType<double>(value);
      }
      return occaUndefined;
    }

    occaType newOccaType(const occa::primitive &value,
                         const int type) {
      switch(type) {
      case occa::c::typeType::int8_   : return newOccaType<int8_t>(value);
      case occa::c::typeType::uint8_  : return newOccaType<uint8_t>(value);
      case occa::c::typeType::int16_  : return newOccaType<int16_t>(value);
      case occa::c::typeType::uint16_ : return newOccaType<uint16_t>(value);
      case occa::c::typeType::int32_  : return newOccaType<int32_t>(value);
      case occa::c::typeType::uint32_ : return newOccaType<uint32_t>(value);
      case occa::c::typeType::int64_  : return newOccaType<int64_t>(value);
      case occa::c::typeType::uint64_ : return newOccaType<uint64_t>(value);
      case occa::c::typeType::float_  : return newOccaType<float>(value);
      case occa::c::typeType::double_ : return newOccaType<double>(value);
      }
      return occaUndefined;
    }

    template <>
    occaType newOccaType(const bool &value) {
      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type  = typeType::bool_;
      oType.bytes = sizeof(int8_t);
      oType.value.int8_ = value;
      return oType;
    }

    template <>
    occaType newOccaType(const int8_t &value) {
      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type  = typeType::int8_;
      oType.bytes = sizeof(int8_t);
      oType.value.int8_ = value;
      return oType;
    }

    template <>
    occaType newOccaType(const uint8_t &value) {
      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type  = typeType::uint8_;
      oType.bytes = sizeof(uint8_t);
      oType.value.uint8_ = value;
      return oType;
    }

    template <>
    occaType newOccaType(const int16_t &value) {
      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type  = typeType::int16_;
      oType.bytes = sizeof(int16_t);
      oType.value.int16_ = value;
      return oType;
    }

    template <>
    occaType newOccaType(const uint16_t &value) {
      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type  = typeType::uint16_;
      oType.bytes = sizeof(uint16_t);
      oType.value.uint16_ = value;
      return oType;
    }

    template <>
    occaType newOccaType(const int32_t &value) {
      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type  = typeType::int32_;
      oType.bytes = sizeof(int32_t);
      oType.value.int32_ = value;
      return oType;
    }

    template <>
    occaType newOccaType(const uint32_t &value) {
      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type  = typeType::uint32_;
      oType.bytes = sizeof(uint32_t);
      oType.value.uint32_ = value;
      return oType;
    }

    template <>
    occaType newOccaType(const int64_t &value) {
      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type  = typeType::int64_;
      oType.bytes = sizeof(int64_t);
      oType.value.int64_ = value;
      return oType;
    }

    template <>
    occaType newOccaType(const uint64_t &value) {
      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type  = typeType::uint64_;
      oType.bytes = sizeof(uint64_t);
      oType.value.uint64_ = value;
      return oType;
    }

    template <>
    occaType newOccaType(const float &value) {
      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type  = typeType::float_;
      oType.bytes = sizeof(float);
      oType.value.float_ = value;
      return oType;
    }

    template <>
    occaType newOccaType(const double &value) {
      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type  = typeType::double_;
      oType.bytes = sizeof(double);
      oType.value.double_ = value;
      return oType;
    }

    occaType newOccaType(occa::device device) {
      occa::modeDevice_t *modeDevice = device.getModeDevice();
      if (!modeDevice) {
        return occaUndefined;
      }

      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type  = typeType::device;
      oType.bytes = sizeof(void*);
      oType.value.ptr = (char*) modeDevice;
      oType.needsFree = false;
      return oType;
    }

    occaType newOccaType(occa::kernel kernel) {
      occa::modeKernel_t *modeKernel = kernel.getModeKernel();
      if (!modeKernel) {
        return occaUndefined;
      }

      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type  = typeType::kernel;
      oType.bytes = sizeof(void*);
      oType.value.ptr = (char*) modeKernel;
      oType.needsFree = false;
      return oType;
    }

    occaType newOccaType(occa::memory memory) {
      occa::modeMemory_t *modeMemory = memory.getModeMemory();
      if (!modeMemory) {
        return occaUndefined;
      }

      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type  = typeType::memory;
      oType.bytes = sizeof(void*);
      oType.value.ptr = (char*) modeMemory;
      oType.needsFree = false;
      return oType;
    }

    occaType newOccaType(occa::stream stream) {
      occa::modeStream_t *modeStream = stream.getModeStream();
      if (!modeStream) {
        return occaUndefined;
      }

      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type  = typeType::stream;
      oType.bytes = sizeof(void*);
      oType.value.ptr = (char*) modeStream;
      oType.needsFree = false;
      return oType;
    }

    occaType newOccaType(occa::streamTag streamTag) {
      occa::modeStreamTag_t *modeStreamTag = streamTag.getModeStreamTag();
      if (!modeStreamTag) {
        return occaUndefined;
      }

      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type  = typeType::streamTag;
      oType.bytes = sizeof(void*);
      oType.value.ptr = (char*) modeStreamTag;
      oType.needsFree = false;
      return oType;
    }

    occaType newOccaType(const occa::json &json,
                         const bool needsFree) {
      if (json.isNull()) {
        return occaNull;
      }
      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type  = typeType::json;
      oType.bytes = sizeof(void*);
      oType.value.ptr = (char*) &json;
      oType.needsFree = needsFree;
      return oType;
    }

    occaType newOccaType(const occa::properties &properties,
                         const bool needsFree) {
      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type  = typeType::properties;
      oType.bytes = sizeof(void*);
      oType.value.ptr = (char*) &properties;
      oType.needsFree = needsFree;
      return oType;
    }

    bool isDefault(occaType value) {
      return (value.type == typeType::default_);
    }

    occa::device device(occaType value) {
      if (occaIsUndefined(value)) {
        return occa::device();
      }
      OCCA_ERROR("Input is not an occaDevice",
                 value.type == typeType::device);
      return occa::device((occa::modeDevice_t*) value.value.ptr);
    }

    occa::kernel kernel(occaType value) {
      if (occaIsUndefined(value)) {
        return occa::kernel();
      }
      OCCA_ERROR("Input is not an occaKernel",
                 value.type == typeType::kernel);
      return occa::kernel((occa::modeKernel_t*) value.value.ptr);
    }

    occa::memory memory(occaType value) {
      if (occaIsUndefined(value)) {
        return occa::memory();
      }
      OCCA_ERROR("Input is not an occaMemory",
                 value.type == typeType::memory);
      return occa::memory((occa::modeMemory_t*) value.value.ptr);
    }

    occa::stream stream(occaType value) {
      if (occaIsUndefined(value)) {
        return occa::stream();
      }
      OCCA_ERROR("Input is not an occaStream",
                 value.type == typeType::stream);
      return occa::stream((occa::modeStream_t*) value.value.ptr);
    }

    occa::streamTag streamTag(occaType value) {
      if (occaIsUndefined(value)) {
        return occa::streamTag();
      }
      OCCA_ERROR("Input is not an occaStreamTag",
                 value.type == typeType::streamTag);
      return occa::streamTag((occa::modeStreamTag_t*) value.value.ptr);
    }

    occa::kernelArg kernelArg(occaType value) {
      OCCA_ERROR("A non-occaType argument was passed",
                 !occaIsUndefined(value));

      occa::kernelArg arg;

      switch (value.type) {
      case occa::c::typeType::ptr: {
        arg.add(value.value.ptr,
                value.bytes,
                false, false);
        break;
      }
      case occa::c::typeType::int8_: {
        return occa::kernelArg(value.value.int8_);
      }
      case occa::c::typeType::uint8_: {
        return occa::kernelArg(value.value.uint8_);
      }
      case occa::c::typeType::int16_: {
        return occa::kernelArg(value.value.int16_);
      }
      case occa::c::typeType::uint16_: {
        return occa::kernelArg(value.value.uint16_);
      }
      case occa::c::typeType::int32_: {
        return occa::kernelArg(value.value.int32_);
      }
      case occa::c::typeType::uint32_: {
        return occa::kernelArg(value.value.uint32_);
      }
      case occa::c::typeType::int64_: {
        return occa::kernelArg(value.value.int64_);
      }
      case occa::c::typeType::uint64_: {
        return occa::kernelArg(value.value.uint64_);
      }
      case occa::c::typeType::float_: {
        return occa::kernelArg(value.value.float_);
      }
      case occa::c::typeType::double_: {
        return occa::kernelArg(value.value.double_);
      }
      case occa::c::typeType::struct_: {
        arg.add(value.value.ptr,
                value.bytes,
                false, false);
        break;
      }
      case occa::c::typeType::string: {
        arg.add(value.value.ptr,
                value.bytes,
                false, false);
        break;
      }
      case occa::c::typeType::memory: {
        return occa::kernelArg(occa::c::memory(value));
      }
      case occa::c::typeType::device:
        OCCA_FORCE_ERROR("Unable to pass an occaDevice as a kernel argument");
      case occa::c::typeType::kernel:
        OCCA_FORCE_ERROR("Unable to pass an occaKernel as a kernel argument");
      case occa::c::typeType::properties:
        OCCA_FORCE_ERROR("Unable to pass an occaProperties as a kernel argument");
      case occa::c::typeType::default_:
        OCCA_FORCE_ERROR("Unable to pass occaDefault as a kernel argument");
      default:
        OCCA_FORCE_ERROR("A non-occaType argument was passed");
      }
      return arg;
    }

    occa::primitive primitive(occaType value) {
      occa::primitive p;

      switch (value.type) {
      case occa::c::typeType::int8_:
        p = value.value.int8_; break;
      case occa::c::typeType::uint8_:
        p = value.value.uint8_; break;
      case occa::c::typeType::int16_:
        p = value.value.int16_; break;
      case occa::c::typeType::uint16_:
        p = value.value.uint16_; break;
      case occa::c::typeType::int32_:
        p = value.value.int32_; break;
      case occa::c::typeType::uint32_:
        p = value.value.uint32_; break;
      case occa::c::typeType::int64_:
        p = value.value.int64_; break;
      case occa::c::typeType::uint64_:
        p = value.value.uint64_; break;
      case occa::c::typeType::float_:
        p = value.value.float_; break;
      case occa::c::typeType::double_:
        p = value.value.double_; break;
      default:
        OCCA_FORCE_ERROR("Invalid value type");
      }

      return p;
    }

    occa::primitive primitive(occaType value,
                              const int type) {
      occa::primitive p = primitive(value);

      switch (type) {
      case occa::c::typeType::int8_: return p.to<int8_t>();
      case occa::c::typeType::uint8_: return p.to<uint8_t>();
      case occa::c::typeType::int16_: return p.to<int16_t>();
      case occa::c::typeType::uint16_: return p.to<uint16_t>();
      case occa::c::typeType::int32_: return p.to<int32_t>();
      case occa::c::typeType::uint32_: return p.to<uint32_t>();
      case occa::c::typeType::int64_: return p.to<int64_t>();
      case occa::c::typeType::uint64_: return p.to<uint64_t>();
      case occa::c::typeType::float_: return p.to<float>();
      case occa::c::typeType::double_: return p.to<double>();
      default:
        OCCA_FORCE_ERROR("Invalid value type");
      }
      return p;
    }

    occa::json& json(occaType value) {
      OCCA_ERROR("Input is not an occaJson",
                 value.type == typeType::json);
      return *((occa::json*) value.value.ptr);
    }

    occa::json inferJson(occaType value) {
      switch (value.type) {
      case occa::c::typeType::bool_:
        return occa::json((bool) value.value.int8_);
      case occa::c::typeType::int8_:
      case occa::c::typeType::uint8_:
      case occa::c::typeType::int16_:
      case occa::c::typeType::uint16_:
      case occa::c::typeType::int32_:
      case occa::c::typeType::uint32_:
      case occa::c::typeType::int64_:
      case occa::c::typeType::uint64_:
      case occa::c::typeType::float_:
      case occa::c::typeType::double_:
        return occa::json(occa::c::primitive(value));
      case occa::c::typeType::string:
        return occa::json((char*) value.value.ptr);
      case occa::c::typeType::json:
        return occa::c::json(value);
      case occa::c::typeType::properties:
        return occa::c::properties(value);
      case occa::c::typeType::ptr:
        if (value.value.ptr == NULL) {
          return occa::json(occa::json::null_);
        }
      default:
        OCCA_FORCE_ERROR("Invalid value type");
        return occa::json();
      }
    }

    occa::properties& properties(occaType value) {
      OCCA_ERROR("Input is not an occaProperties",
                 value.type == typeType::properties);
      return *((occa::properties*) value.value.ptr);
    }

    const occa::properties& constProperties(occaType value) {
      OCCA_ERROR("Input is not an occaProperties",
                 value.type == typeType::properties);
      return *((const occa::properties*) value.value.ptr);
    }
  }
}

OCCA_START_EXTERN_C

//---[ Type Flags ]---------------------
const int OCCA_UNDEFINED  = occa::c::typeType::undefined;
const int OCCA_DEFAULT    = occa::c::typeType::default_;

const int OCCA_PTR        = occa::c::typeType::ptr;

const int OCCA_BOOL       = occa::c::typeType::bool_;

const int OCCA_INT8       = occa::c::typeType::int8_;
const int OCCA_UINT8      = occa::c::typeType::uint8_;
const int OCCA_INT16      = occa::c::typeType::int16_;
const int OCCA_UINT16     = occa::c::typeType::uint16_;
const int OCCA_INT32      = occa::c::typeType::int32_;
const int OCCA_UINT32     = occa::c::typeType::uint32_;
const int OCCA_INT64      = occa::c::typeType::int64_;
const int OCCA_UINT64     = occa::c::typeType::uint64_;
const int OCCA_FLOAT      = occa::c::typeType::float_;
const int OCCA_DOUBLE     = occa::c::typeType::double_;

const int OCCA_STRUCT     = occa::c::typeType::struct_;
const int OCCA_STRING     = occa::c::typeType::string;

const int OCCA_DEVICE     = occa::c::typeType::device;
const int OCCA_KERNEL     = occa::c::typeType::kernel;
const int OCCA_MEMORY     = occa::c::typeType::memory;
const int OCCA_STREAM     = occa::c::typeType::stream;
const int OCCA_STREAMTAG  = occa::c::typeType::streamTag;

const int OCCA_JSON       = occa::c::typeType::json;
const int OCCA_PROPERTIES = occa::c::typeType::properties;
//======================================

//---[ Globals & Flags ]----------------
const occaType occaNull       = occa::c::newOccaType((void*) NULL);
const occaType occaUndefined  = occa::c::undefinedOccaType();
const occaType occaDefault    = occa::c::defaultOccaType();
const occaUDim_t occaAllBytes = -1;
//======================================

//-----[ Known Types ]------------------
OCCA_LFUNC int OCCA_RFUNC occaIsUndefined(occaType value) {
  return ((value.magicHeader == OCCA_C_TYPE_UNDEFINED_HEADER) ||
          (value.magicHeader != OCCA_C_TYPE_MAGIC_HEADER));
}

OCCA_LFUNC int OCCA_RFUNC occaIsDefault(occaType value) {
  return (value.type == occa::c::typeType::default_);
}

OCCA_LFUNC occaType OCCA_RFUNC occaPtr(void *value) {
  return occa::c::newOccaType(value);
}

OCCA_LFUNC occaType OCCA_RFUNC occaBool(int value) {
  return occa::c::newOccaType((bool) value);
}

OCCA_LFUNC occaType OCCA_RFUNC occaInt8(int8_t value) {
  return occa::c::newOccaType(value);
}

OCCA_LFUNC occaType OCCA_RFUNC occaUInt8(uint8_t value) {
  return occa::c::newOccaType(value);
}

OCCA_LFUNC occaType OCCA_RFUNC occaInt16(int16_t value) {
  return occa::c::newOccaType(value);
}

OCCA_LFUNC occaType OCCA_RFUNC occaUInt16(uint16_t value) {
  return occa::c::newOccaType(value);
}

OCCA_LFUNC occaType OCCA_RFUNC occaInt32(int32_t value) {
  return occa::c::newOccaType(value);
}

OCCA_LFUNC occaType OCCA_RFUNC occaUInt32(uint32_t value) {
  return occa::c::newOccaType(value);
}

OCCA_LFUNC occaType OCCA_RFUNC occaInt64(int64_t value) {
  return occa::c::newOccaType(value);
}

OCCA_LFUNC occaType OCCA_RFUNC occaUInt64(uint64_t value) {
  return occa::c::newOccaType(value);
}
//======================================

//-----[ Ambiguous Types ]--------------
OCCA_LFUNC occaType OCCA_RFUNC occaChar(char value) {
  return occa::c::newOccaIntType<char>(false, value);
}

OCCA_LFUNC occaType OCCA_RFUNC occaUChar(unsigned char value) {
  return occa::c::newOccaIntType<char>(true, value);
}

OCCA_LFUNC occaType OCCA_RFUNC occaShort(short value) {
  return occa::c::newOccaIntType<short>(false, value);
}

OCCA_LFUNC occaType OCCA_RFUNC occaUShort(unsigned short value) {
  return occa::c::newOccaIntType<short>(true, value);
}

OCCA_LFUNC occaType OCCA_RFUNC occaInt(int value) {
  return occa::c::newOccaIntType<int>(false, value);
}

OCCA_LFUNC occaType OCCA_RFUNC occaUInt(unsigned int value) {
  return occa::c::newOccaIntType<int>(true, value);
}

OCCA_LFUNC occaType OCCA_RFUNC occaLong(long value) {
  return occa::c::newOccaIntType<long>(false, value);
}

OCCA_LFUNC occaType OCCA_RFUNC occaULong(unsigned long value) {
  return occa::c::newOccaIntType<long>(true, value);
}

OCCA_LFUNC occaType OCCA_RFUNC occaFloat(float value) {
  return occa::c::newOccaType(value);
}

OCCA_LFUNC occaType OCCA_RFUNC occaDouble(double value) {
  return occa::c::newOccaType(value);
}

OCCA_LFUNC occaType OCCA_RFUNC occaStruct(void *value,
                                          occaUDim_t bytes) {
  occaType oType;
  oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
  oType.type  = occa::c::typeType::struct_;
  oType.bytes = bytes;
  oType.value.ptr = (char*) value;
  oType.needsFree = false;
  return oType;
}

OCCA_LFUNC occaType OCCA_RFUNC occaString(const char *str) {
  occaType oType;
  oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
  oType.type  = occa::c::typeType::string;
  oType.bytes = strlen(str);
  oType.value.ptr = const_cast<char*>(str);
  oType.needsFree = false;
  return oType;
}
//======================================

OCCA_LFUNC void OCCA_RFUNC occaFree(occaType value) {
  if (occaIsUndefined(value)) {
    return;
  }
  switch (value.type) {
  case occa::c::typeType::device: {
    occa::c::device(value).free();
    break;
  }
  case occa::c::typeType::kernel: {
    occa::c::kernel(value).free();
    break;
  }
  case occa::c::typeType::memory: {
    occa::c::memory(value).free();
    break;
  }
  case occa::c::typeType::stream: {
    occa::c::stream(value).free();
    break;
  }
  case occa::c::typeType::streamTag: {
    occa::c::streamTag(value).free();
    break;
  }
  case occa::c::typeType::json: {
    if (value.needsFree) {
      delete &occa::c::json(value);
    }
    break;
  }
  case occa::c::typeType::properties: {
    if (value.needsFree) {
      delete &occa::c::properties(value);
    }
    break;
  }}
  value.magicHeader = occaUndefined.magicHeader;
}

OCCA_END_EXTERN_C
