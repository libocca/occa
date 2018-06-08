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
    occaType defaultOccaType() {
      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type = occa::c::typeType::none;
      oType.value.ptr = NULL;
      return oType;
    }

    occaType newOccaType(void *value) {
      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type  = typeType::ptr;
      oType.bytes = sizeof(void*);
      oType.value.ptr = (char*) value;
      return oType;
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
      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type  = typeType::device;
      oType.bytes = sizeof(void*);
      oType.value.ptr = (char*) device.getDHandle();
      return oType;
    }

    occaType newOccaType(occa::kernel kernel) {
      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type  = typeType::kernel;
      oType.bytes = sizeof(void*);
      oType.value.ptr = (char*) kernel.getKHandle();
      return oType;
    }

    occaType newOccaType(occa::memory memory) {
      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type  = typeType::memory;
      oType.bytes = sizeof(void*);
      oType.value.ptr = (char*) memory.getMHandle();
      return oType;
    }

    occaType newOccaType(occa::properties &properties) {
      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type  = typeType::properties;
      oType.bytes = sizeof(void*);
      oType.value.ptr = (char*) &properties;
      return oType;
    }

    occaStream newOccaType(occa::stream value) {
      occaStream stream;
      stream.device = newOccaType(occa::device(value.dHandle));
      stream.handle = value.handle;
      return stream;
    }

    occaStreamTag newOccaType(occa::streamTag value) {
      occaStreamTag tag;
      tag.tagTime = value.tagTime;
      tag.handle  = value.handle;
      return tag;
    }

    bool isDefault(occaType value) {
      return ((value.type == typeType::none) &&
              (value.value.ptr == NULL));
    }

    occa::device device(occaType value) {
      OCCA_ERROR("Input is not an occaDevice",
                 value.type == typeType::device);
      return occa::device((occa::device_v*) value.value.ptr);
    }

    occa::kernel kernel(occaType value) {
      OCCA_ERROR("Input is not an occaKernel",
                 value.type == typeType::kernel);
      return occa::kernel((occa::kernel_v*) value.value.ptr);
    }

    occa::memory memory(occaType value) {
      OCCA_ERROR("Input is not an occaMemory",
                 value.type == typeType::memory);
      return occa::memory((occa::memory_v*) value.value.ptr);
    }

    occa::properties& properties(occaType value) {
      OCCA_ERROR("Input is not an occaProperties",
                 value.type == typeType::properties);
      return *((occa::properties*) value.value.ptr);
    }

    occa::stream stream(occaStream value) {
      return occa::stream((occa::device_v*) value.device.value.ptr,
                          value.handle);
    }

    occa::streamTag streamTag(occaStreamTag value) {
      return occa::streamTag(value.tagTime,
                             value.handle);
    }
  }
}

OCCA_START_EXTERN_C

//---[ Type Flags ]---------------------
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
const int OCCA_PROPERTIES = occa::c::typeType::properties;
//======================================

//---[ Globals & Flags ]----------------
const occaType occaDefault    = occa::c::defaultOccaType();
const occaUDim_t occaAllBytes = -1;
//======================================

//-----[ Known Types ]------------------
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
  return occa::c::newOccaType(value);
}

OCCA_LFUNC occaType OCCA_RFUNC occaUChar(unsigned char value) {
  return occa::c::newOccaType(value);
}

OCCA_LFUNC occaType OCCA_RFUNC occaShort(short value) {
  return occa::c::newOccaType(value);
}

OCCA_LFUNC occaType OCCA_RFUNC occaUShort(unsigned short value) {
  return occa::c::newOccaType(value);
}

OCCA_LFUNC occaType OCCA_RFUNC occaInt(int value) {
  return occa::c::newOccaType(value);
}

OCCA_LFUNC occaType OCCA_RFUNC occaUInt(unsigned int value) {
  return occa::c::newOccaType(value);
}

OCCA_LFUNC occaType OCCA_RFUNC occaLong(long value) {
  return occa::c::newOccaType(value);
}

OCCA_LFUNC occaType OCCA_RFUNC occaULong(unsigned long value) {
  return occa::c::newOccaType(value);
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
  return oType;
}

OCCA_LFUNC occaType OCCA_RFUNC occaString(const char *str) {
  occaType oType;
  oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
  oType.type  = occa::c::typeType::string;
  oType.bytes = strlen(str);
  oType.value.ptr = const_cast<char*>(str);
  return oType;
}
//======================================

OCCA_LFUNC void OCCA_RFUNC occaFree(occaType value) {
  switch (value.type) {
  case occa::c::typeType::device: {
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
  case occa::c::typeType::properties: {
    delete &occa::c::properties(value);
    break;
  }}
}

OCCA_LFUNC void OCCA_RFUNC occaFreeStream(occaStream value) {
  occa::c::stream(value).free();
}

OCCA_END_EXTERN_C
