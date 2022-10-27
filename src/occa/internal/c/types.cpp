#include <cstring>

#include <occa/internal/c/types.hpp>
#include <occa/internal/core/memory.hpp>
#include <occa/internal/core/memoryPool.hpp>

namespace occa {
  namespace c {
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

    occaType nullOccaType() {
      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type = occa::c::typeType::null_;
      oType.value.ptr = NULL;
      oType.needsFree = false;
      return oType;
    }

    occaType newOccaType(void *value) {
      return newOccaType((const void*) value);
    }

    occaType newOccaType(const void *value) {
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
      oType.needsFree = false;
      return oType;
    }

    template <>
    occaType newOccaType(const int8_t &value) {
      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type  = typeType::int8_;
      oType.bytes = sizeof(int8_t);
      oType.value.int8_ = value;
      oType.needsFree = false;
      return oType;
    }

    template <>
    occaType newOccaType(const uint8_t &value) {
      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type  = typeType::uint8_;
      oType.bytes = sizeof(uint8_t);
      oType.value.uint8_ = value;
      oType.needsFree = false;
      return oType;
    }

    template <>
    occaType newOccaType(const int16_t &value) {
      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type  = typeType::int16_;
      oType.bytes = sizeof(int16_t);
      oType.value.int16_ = value;
      oType.needsFree = false;
      return oType;
    }

    template <>
    occaType newOccaType(const uint16_t &value) {
      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type  = typeType::uint16_;
      oType.bytes = sizeof(uint16_t);
      oType.value.uint16_ = value;
      oType.needsFree = false;
      return oType;
    }

    template <>
    occaType newOccaType(const int32_t &value) {
      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type  = typeType::int32_;
      oType.bytes = sizeof(int32_t);
      oType.value.int32_ = value;
      oType.needsFree = false;
      return oType;
    }

    template <>
    occaType newOccaType(const uint32_t &value) {
      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type  = typeType::uint32_;
      oType.bytes = sizeof(uint32_t);
      oType.value.uint32_ = value;
      oType.needsFree = false;
      return oType;
    }

    template <>
    occaType newOccaType(const int64_t &value) {
      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type  = typeType::int64_;
      oType.bytes = sizeof(int64_t);
      oType.value.int64_ = value;
      oType.needsFree = false;
      return oType;
    }

    template <>
    occaType newOccaType(const uint64_t &value) {
      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type  = typeType::uint64_;
      oType.bytes = sizeof(uint64_t);
      oType.value.uint64_ = value;
      oType.needsFree = false;
      return oType;
    }

    template <>
    occaType newOccaType(const float &value) {
      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type  = typeType::float_;
      oType.bytes = sizeof(float);
      oType.value.float_ = value;
      oType.needsFree = false;
      return oType;
    }

    template <>
    occaType newOccaType(const double &value) {
      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type  = typeType::double_;
      oType.bytes = sizeof(double);
      oType.value.double_ = value;
      oType.needsFree = false;
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

    occaType newOccaType(occa::experimental::memoryPool memoryPool) {
      occa::modeMemoryPool_t *modeMemoryPool = memoryPool.getModeMemoryPool();
      if (!modeMemoryPool) {
        return occaUndefined;
      }

      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type  = typeType::memoryPool;
      oType.bytes = sizeof(void*);
      oType.value.ptr = (char*) modeMemoryPool;
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

    occaType newOccaType(const occa::kernelBuilder &kernelBuilder) {
      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type  = typeType::kernelBuilder;
      oType.bytes = sizeof(void*);
      oType.value.ptr = (char*) &kernelBuilder;
      oType.needsFree = false;
      return oType;
    }

    occaType newOccaType(const occa::dtype_t &dtype) {
      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type  = typeType::dtype;
      oType.bytes = sizeof(void*);
      oType.value.ptr = (char*) &dtype;
      oType.needsFree = true;
      return oType;
    }

    occaType newOccaType(const occa::scope &scope) {
      occaType oType;
      oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
      oType.type  = typeType::scope;
      oType.bytes = sizeof(void*);
      oType.value.ptr = (char*) &scope;
      oType.needsFree = true;
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
      return nullOccaType();
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

    occa::kernelBuilder kernelBuilder(occaType value) {
      OCCA_ERROR("Input is not an occaKernelBuilder",
                 value.type == typeType::kernelBuilder);
      return *((occa::kernelBuilder*) value.value.ptr);
    }

    occa::memory memory(occaType value) {
      if (occaIsUndefined(value)) {
        return occa::memory();
      }
      OCCA_ERROR("Input is not an occaMemory",
                 value.type == typeType::memory);
      return occa::memory((occa::modeMemory_t*) value.value.ptr);
    }

    occa::experimental::memoryPool memoryPool(occaType value) {
      if (occaIsUndefined(value)) {
        return occa::experimental::memoryPool();
      }
      OCCA_ERROR("Input is not an occaMemoryPool",
                 value.type == typeType::memoryPool);
      return occa::experimental::memoryPool((occa::modeMemoryPool_t*) value.value.ptr);
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
          arg.addPointer(value.value.ptr,
                         value.bytes);
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
          arg.addPointer(value.value.ptr,
                         value.bytes);
          break;
        }
        case occa::c::typeType::string: {
          arg.addPointer(value.value.ptr,
                         value.bytes);
          break;
        }
        case occa::c::typeType::memory: {
          return occa::kernelArg(occa::c::memory(value));
        }
        case occa::c::typeType::null_: {
          return occa::kernelArg(occa::null);
        }
        default:
          OCCA_FORCE_ERROR("An invalid occaType or non-occaType argument was passed");
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

    occa::dtype_t& dtype(occaType value) {
      OCCA_ERROR("Input is not an occaDtype",
                 value.type == typeType::dtype);
      return *((occa::dtype_t*) value.value.ptr);
    }

    occa::scope& scope(occaType value) {
      OCCA_ERROR("Input is not an occaScope",
                 value.type == typeType::scope);
      return *((occa::scope*) value.value.ptr);
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
        case occa::c::typeType::null_:
          return occa::json(occa::json::null_);
        case occa::c::typeType::ptr:
          if (value.value.ptr == NULL) {
            return occa::json(occa::json::null_);
          }
          /* FALLTHRU */
        default:
          OCCA_FORCE_ERROR("Invalid value type");
          return occa::json();
      }
    }

    occa::dtype_t getDtype(occaType value) {
      switch (value.type) {
        case occa::c::typeType::bool_:
          return dtype::bool_;
        case occa::c::typeType::int8_:
          return dtype::int8;
        case occa::c::typeType::uint8_:
          return dtype::uint8;
        case occa::c::typeType::int16_:
          return dtype::int16;
        case occa::c::typeType::uint16_:
          return dtype::uint16;
        case occa::c::typeType::int32_:
          return dtype::int32;
        case occa::c::typeType::uint32_:
          return dtype::uint32;
        case occa::c::typeType::int64_:
          return dtype::int64;
        case occa::c::typeType::uint64_:
          return dtype::uint64;
        case occa::c::typeType::float_:
          return dtype::float_;
        case occa::c::typeType::double_:
          return dtype::double_;
        case occa::c::typeType::memory:
          return occa::c::memory(value).dtype();
        case occa::c::typeType::null_:
          return dtype::void_;
        default:
          OCCA_FORCE_ERROR("Invalid value type");
          return dtype::none;
      }
    }
  }
}

OCCA_START_EXTERN_C

//---[ Type Flags ]---------------------
const int OCCA_UNDEFINED     = occa::c::typeType::undefined;
const int OCCA_DEFAULT       = occa::c::typeType::default_;
const int OCCA_NULL          = occa::c::typeType::null_;

const int OCCA_PTR           = occa::c::typeType::ptr;

const int OCCA_BOOL          = occa::c::typeType::bool_;

const int OCCA_INT8          = occa::c::typeType::int8_;
const int OCCA_UINT8         = occa::c::typeType::uint8_;
const int OCCA_INT16         = occa::c::typeType::int16_;
const int OCCA_UINT16        = occa::c::typeType::uint16_;
const int OCCA_INT32         = occa::c::typeType::int32_;
const int OCCA_UINT32        = occa::c::typeType::uint32_;
const int OCCA_INT64         = occa::c::typeType::int64_;
const int OCCA_UINT64        = occa::c::typeType::uint64_;
const int OCCA_FLOAT         = occa::c::typeType::float_;
const int OCCA_DOUBLE        = occa::c::typeType::double_;

const int OCCA_STRUCT        = occa::c::typeType::struct_;
const int OCCA_STRING        = occa::c::typeType::string;

const int OCCA_DEVICE        = occa::c::typeType::device;
const int OCCA_KERNEL        = occa::c::typeType::kernel;
const int OCCA_KERNELBUILDER = occa::c::typeType::kernelBuilder;
const int OCCA_MEMORY        = occa::c::typeType::memory;
const int OCCA_MEMORYPOOL    = occa::c::typeType::memoryPool;
const int OCCA_STREAM        = occa::c::typeType::stream;
const int OCCA_STREAMTAG     = occa::c::typeType::streamTag;

const int OCCA_DTYPE         = occa::c::typeType::dtype;
const int OCCA_SCOPE         = occa::c::typeType::scope;
const int OCCA_JSON          = occa::c::typeType::json;
//======================================

//---[ Globals & Flags ]----------------
const occaType occaUndefined  = occa::c::undefinedOccaType();
const occaType occaDefault    = occa::c::defaultOccaType();
const occaType occaNull       = occa::c::nullOccaType();
const occaType occaTrue       = occa::c::newOccaType(true);
const occaType occaFalse      = occa::c::newOccaType(false);
const occaUDim_t occaAllBytes = occa::UDIM_DEFAULT;
//======================================

//-----[ Known Types ]------------------
bool occaIsUndefined(occaType value) {
  return ((value.magicHeader == OCCA_C_TYPE_UNDEFINED_HEADER) ||
          (value.magicHeader != OCCA_C_TYPE_MAGIC_HEADER));
}

bool occaIsNull(occaType value) {
  return (value.type == occa::c::typeType::null_);
}

bool occaIsDefault(occaType value) {
  return (value.type == occa::c::typeType::default_);
}

occaType occaPtr(const void *value) {
  return occa::c::newOccaType(value);
}

occaType occaBool(bool value) {
  return occa::c::newOccaType(value);
}

occaType occaInt8(int8_t value) {
  return occa::c::newOccaType(value);
}

occaType occaUInt8(uint8_t value) {
  return occa::c::newOccaType(value);
}

occaType occaInt16(int16_t value) {
  return occa::c::newOccaType(value);
}

occaType occaUInt16(uint16_t value) {
  return occa::c::newOccaType(value);
}

occaType occaInt32(int32_t value) {
  return occa::c::newOccaType(value);
}

occaType occaUInt32(uint32_t value) {
  return occa::c::newOccaType(value);
}

occaType occaInt64(int64_t value) {
  return occa::c::newOccaType(value);
}

occaType occaUInt64(uint64_t value) {
  return occa::c::newOccaType(value);
}
//======================================

//-----[ Ambiguous Types ]--------------
occaType occaChar(char value) {
  return occa::c::newOccaIntType<char>(false, value);
}

occaType occaUChar(unsigned char value) {
  return occa::c::newOccaIntType<char>(true, value);
}

occaType occaShort(short value) {
  return occa::c::newOccaIntType<short>(false, value);
}

occaType occaUShort(unsigned short value) {
  return occa::c::newOccaIntType<short>(true, value);
}

occaType occaInt(int value) {
  return occa::c::newOccaIntType<int>(false, value);
}

occaType occaUInt(unsigned int value) {
  return occa::c::newOccaIntType<int>(true, value);
}

occaType occaLong(long value) {
  return occa::c::newOccaIntType<long>(false, value);
}

occaType occaULong(unsigned long value) {
  return occa::c::newOccaIntType<long>(true, value);
}

occaType occaFloat(float value) {
  return occa::c::newOccaType(value);
}

occaType occaDouble(double value) {
  return occa::c::newOccaType(value);
}

occaType occaStruct(const void *value,
                    occaUDim_t bytes) {
  occaType oType;
  oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
  oType.type  = occa::c::typeType::struct_;
  oType.bytes = bytes;
  oType.value.ptr = (char*) value;
  oType.needsFree = false;
  return oType;
}

occaType occaString(const char *str) {
  occaType oType;
  oType.magicHeader = OCCA_C_TYPE_MAGIC_HEADER;
  oType.type  = occa::c::typeType::string;
  oType.bytes = strlen(str);
  oType.value.ptr = const_cast<char*>(str);
  oType.needsFree = false;
  return oType;
}
//======================================

void occaFree(occaType *value) {
  occaType &valueRef = *value;

  if (occaIsUndefined(valueRef)) {
    return;
  }
  switch (valueRef.type) {
    case occa::c::typeType::device: {
      occa::c::device(valueRef).free();
      break;
    }
    case occa::c::typeType::kernel: {
      occa::c::kernel(valueRef).free();
      break;
    }
    case occa::c::typeType::kernelBuilder: {
      occa::c::kernelBuilder(valueRef).free();
      break;
    }
    case occa::c::typeType::memory: {
      occa::c::memory(valueRef).free();
      break;
    }
    case occa::c::typeType::memoryPool: {
      occa::c::memoryPool(valueRef).free();
      break;
    }
    case occa::c::typeType::stream: {
      occa::c::stream(valueRef).free();
      break;
    }
    case occa::c::typeType::streamTag: {
      occa::c::streamTag(valueRef).free();
      break;
    }
    case occa::c::typeType::dtype: {
      delete &occa::c::dtype(valueRef);
      break;
    }
    case occa::c::typeType::scope: {
      delete &occa::c::scope(valueRef);
      break;
    }
    case occa::c::typeType::json: {
      if (valueRef.needsFree) {
        delete &occa::c::json(valueRef);
      }
      break;
    }}
  valueRef.magicHeader = occaUndefined.magicHeader;
}

void occaPrintTypeInfo(occaType value) {
  occa::json info({
    {"type", "undefined"}
  });

  if (!occaIsUndefined(value)) {
    switch (value.type) {
      case occa::c::typeType::default_:
        info["type"] = "default";
        break;
      case occa::c::typeType::null_:
        info["type"]  = "ptr";
        info["value"] = "NULL";

        break;
      case occa::c::typeType::ptr:
        info["type"]  = "ptr";
        info["value"] = (void*) value.value.ptr;

        break;
      case occa::c::typeType::bool_:
        info["type"]  = "bool";
        info["value"] = (bool) value.value.int8_;

        break;
      case occa::c::typeType::int8_:
        info["type"]  = "int8";
        info["value"] = value.value.int8_;

        break;
      case occa::c::typeType::uint8_:
        info["type"]  = "uint8";
        info["value"] = value.value.uint8_;

        break;
      case occa::c::typeType::int16_:
        info["type"]  = "int16";
        info["value"] = value.value.int16_;

        break;
      case occa::c::typeType::uint16_:
        info["type"]  = "uint16";
        info["value"] = value.value.uint16_;

        break;
      case occa::c::typeType::int32_:
        info["type"]  = "int32";
        info["value"] = value.value.int32_;

        break;
      case occa::c::typeType::uint32_:
        info["type"]  = "uint32";
        info["value"] = value.value.uint32_;

        break;
      case occa::c::typeType::int64_:
        info["type"]  = "int64";
        info["value"] = value.value.int64_;

        break;
      case occa::c::typeType::uint64_:
        info["type"]  = "uint64";
        info["value"] = value.value.uint64_;

        break;
      case occa::c::typeType::float_:
        info["type"]  = "float";
        info["value"] = value.value.float_;

        break;
      case occa::c::typeType::double_:
        info["type"]  = "double";
        info["value"] = value.value.double_;

        break;
      case occa::c::typeType::struct_:
        info["type"]  = "struct";
        info["value"] = (void*) value.value.ptr;
        info["bytes"] = value.bytes;

        break;
      case occa::c::typeType::string:
        info["type"]  = "string";
        info["value"] = std::string(value.value.ptr);

        break;
      case occa::c::typeType::device: {
        info["type"]  = "device";
        info["value"] = (void*) value.value.ptr;

        occa::device device = occa::c::device(value);
        if (device.isInitialized()) {
        info["mode"]  = device.mode();
        info["props"] = device.properties();
        } else {
          info["initialized"] = false;
        }

        break;
      }
      case occa::c::typeType::kernel: {
        info["type"]  = "kernel";
        info["value"] = (void*) value.value.ptr;

        occa::kernel kernel = occa::c::kernel(value);
        if (kernel.isInitialized()) {
          info["mode"]  = kernel.mode();
          info["props"] = kernel.properties();
          info["name"]  = kernel.name();
        } else {
          info["initialized"] = false;
        }

        break;
      }
      case occa::c::typeType::kernelBuilder: {
        info["type"]  = "kernelBuilder";
        info["value"] = (void*) value.value.ptr;

        occa::kernelBuilder kernelBuilder = occa::c::kernelBuilder(value);
        if (kernelBuilder.isInitialized()) {
          info["kernel_name"] = kernelBuilder.getKernelName();
        } else {
          info["initialized"] = false;
        }

        break;
      }
      case occa::c::typeType::memory: {
        info["type"]  = "memory";
        info["value"] = (void*) value.value.ptr;

        occa::memory mem = occa::c::memory(value);
        if (mem.isInitialized()) {
          info["mode"]   = mem.mode();
          info["props"]  = mem.properties();
          info["dtype"]  = mem.dtype().toJson();
          info["length"] = mem.length();
          info["size"]   = mem.size();
        } else {
          info["initialized"] = false;
        }

        break;
      }
      case occa::c::typeType::memoryPool: {
        info["type"]  = "memoryPool";
        info["value"] = (void*) value.value.ptr;

        occa::experimental::memoryPool memPool = occa::c::memoryPool(value);
        if (memPool.isInitialized()) {
          info["mode"]   = memPool.mode();
          info["props"]  = memPool.properties();
        } else {
          info["initialized"] = false;
        }

        break;
      }
      case occa::c::typeType::stream: {
        info["type"]  = "stream";
        info["value"] = (void*) value.value.ptr;

        occa::stream stream = occa::c::stream(value);
        if (stream.isInitialized()) {
          info["mode"]  = stream.getDevice().mode();
        } else {
          info["initialized"] = false;
        }

        break;
      }
      case occa::c::typeType::streamTag: {
        info["type"]  = "streamTag";
        info["value"] = (void*) value.value.ptr;

        occa::streamTag streamTag = occa::c::streamTag(value);
        if (streamTag.isInitialized()) {
          info["mode"]  = streamTag.getDevice().mode();
        } else {
          info["initialized"] = false;
        }

        break;
      }
      case occa::c::typeType::dtype: {
        info["type"]  = "dtype";
        info["value"] = (void*) value.value.ptr;
        info["dtype"] = occa::c::dtype(value).toJson();

        break;
      }
      case occa::c::typeType::scope: {
        occa::scope &scope = occa::c::scope(value);

        info["type"]  = "scope";
        info["value"] = (void*) value.value.ptr;
        info["props"] = scope.props;

        occa::json args = info["args"].asArray();
        for (auto &arg : scope.args) {
          args += occa::json({
            {"name", arg.name},
            {"dtype", arg.dtype.toJson()},
            {"is_const", arg.isConst}
          });
        }

        break;
      }
      case occa::c::typeType::json:
        info["type"]  = "json";
        info["value"] = occa::c::json(value);
    }
  }

  std::cout << info << '\n';
}

OCCA_END_EXTERN_C
