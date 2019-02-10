#include <occa/c/types.hpp>
#include <occa/c/dtype.h>

OCCA_START_EXTERN_C

//-----[ Methods ]----------------------
OCCA_LFUNC occaDtype OCCA_RFUNC occaCreateDtype(const char *name,
                                                const int bytes) {
  return occa::c::newOccaType(
    *(new occa::dtype_t(name, bytes))
  );
}

OCCA_LFUNC occaDtype OCCA_RFUNC occaCreateGlobalDtype(const char *name,
                                                      const int bytes) {
  return occa::c::newOccaType(
    *(new occa::dtype_t(name, bytes, true))
  );
}

OCCA_LFUNC occaDtype OCCA_RFUNC occaCreateDtypeTuple(occaDtype dtype,
                                                     const int size) {
  return occa::c::newOccaType(
    *(new occa::dtype_t(
        occa::dtype_t::tuple(occa::c::dtype(dtype), size)
      ))
  );
}

OCCA_LFUNC const char* OCCA_RFUNC occaDtypeName(occaDtype dtype) {
  return occa::c::dtype(dtype).name().c_str();
}

OCCA_LFUNC int OCCA_RFUNC occaDtypeBytes(occaDtype dtype) {
  return occa::c::dtype(dtype).bytes();
}

OCCA_LFUNC void OCCA_RFUNC occaDtypeAddField(occaDtype dtype,
                                             const char *field,
                                             occaDtype fieldType) {
  occa::dtype_t &dtype_ = occa::c::dtype(dtype);
  dtype_.addField(field,
                  occa::c::dtype(fieldType));
}

OCCA_LFUNC int OCCA_RFUNC occaDtypesAreEqual(occaDtype a,
                                             occaDtype b) {
  return (occa::c::dtype(a) == occa::c::dtype(b));
}

OCCA_LFUNC int OCCA_RFUNC occaDtypesMatch(occaDtype a,
                                          occaDtype b) {
  return occa::c::dtype(a).matches(occa::c::dtype(b));
}

OCCA_LFUNC occaDtype OCCA_RFUNC occaDtypeFromJson(occaJson json) {
  return occa::c::newOccaType(
    *(new occa::dtype_t(
        occa::dtype_t::fromJson(occa::c::json(json))
      ))
  );
}
OCCA_LFUNC occaDtype OCCA_RFUNC occaDtypeFromJsonString(const char *str) {
  return occa::c::newOccaType(
    *(new occa::dtype_t(
        occa::dtype_t::fromJson(std::string(str))
      ))
  );
}

OCCA_LFUNC occaJson OCCA_RFUNC occaDtypeToJson(occaDtype dtype) {
  occa::dtype_t &dtype_ = occa::c::dtype(dtype);
  return occa::c::newOccaType(
    *(new occa::json(dtype_.toJson())),
    true
  );
}
//======================================

//-----[ Builtins ]---------------------
occaDtype occaDtypeNone = occa::c::newOccaType(occa::dtype::none);

occaDtype occaDtypeVoid = occa::c::newOccaType(occa::dtype::void_);
occaDtype occaDtypeByte = occa::c::newOccaType(occa::dtype::byte);

occaDtype occaDtypeBool   = occa::c::newOccaType(occa::dtype::bool_);
occaDtype occaDtypeChar   = occa::c::newOccaType(occa::dtype::char_);
occaDtype occaDtypeShort  = occa::c::newOccaType(occa::dtype::short_);
occaDtype occaDtypeInt    = occa::c::newOccaType(occa::dtype::int_);
occaDtype occaDtypeLong   = occa::c::newOccaType(occa::dtype::long_);
occaDtype occaDtypeFloat  = occa::c::newOccaType(occa::dtype::float_);
occaDtype occaDtypeDouble = occa::c::newOccaType(occa::dtype::double_);

occaDtype occaDtypeInt8    = occa::c::newOccaType(occa::dtype::int8);
occaDtype occaDtypeUint8   = occa::c::newOccaType(occa::dtype::uint8);
occaDtype occaDtypeInt16   = occa::c::newOccaType(occa::dtype::int16);
occaDtype occaDtypeUint16  = occa::c::newOccaType(occa::dtype::uint16);
occaDtype occaDtypeInt32   = occa::c::newOccaType(occa::dtype::int32);
occaDtype occaDtypeUint32  = occa::c::newOccaType(occa::dtype::uint32);
occaDtype occaDtypeInt64   = occa::c::newOccaType(occa::dtype::int64);
occaDtype occaDtypeUint64  = occa::c::newOccaType(occa::dtype::uint64);
occaDtype occaDtypeFloat32 = occa::c::newOccaType(occa::dtype::float32);
occaDtype occaDtypeFloat64 = occa::c::newOccaType(occa::dtype::float64);

// OKL Primitives
occaDtype occaDtypeUchar2 = occa::c::newOccaType(occa::dtype::uchar2);
occaDtype occaDtypeUchar3 = occa::c::newOccaType(occa::dtype::uchar3);
occaDtype occaDtypeUchar4 = occa::c::newOccaType(occa::dtype::uchar4);

occaDtype occaDtypeChar2 = occa::c::newOccaType(occa::dtype::char2);
occaDtype occaDtypeChar3 = occa::c::newOccaType(occa::dtype::char3);
occaDtype occaDtypeChar4 = occa::c::newOccaType(occa::dtype::char4);

occaDtype occaDtypeUshort2 = occa::c::newOccaType(occa::dtype::ushort2);
occaDtype occaDtypeUshort3 = occa::c::newOccaType(occa::dtype::ushort3);
occaDtype occaDtypeUshort4 = occa::c::newOccaType(occa::dtype::ushort4);

occaDtype occaDtypeShort2 = occa::c::newOccaType(occa::dtype::short2);
occaDtype occaDtypeShort3 = occa::c::newOccaType(occa::dtype::short3);
occaDtype occaDtypeShort4 = occa::c::newOccaType(occa::dtype::short4);

occaDtype occaDtypeUint2 = occa::c::newOccaType(occa::dtype::uint2);
occaDtype occaDtypeUint3 = occa::c::newOccaType(occa::dtype::uint3);
occaDtype occaDtypeUint4 = occa::c::newOccaType(occa::dtype::uint4);

occaDtype occaDtypeInt2 = occa::c::newOccaType(occa::dtype::int2);
occaDtype occaDtypeInt3 = occa::c::newOccaType(occa::dtype::int3);
occaDtype occaDtypeInt4 = occa::c::newOccaType(occa::dtype::int4);

occaDtype occaDtypeUlong2 = occa::c::newOccaType(occa::dtype::ulong2);
occaDtype occaDtypeUlong3 = occa::c::newOccaType(occa::dtype::ulong3);
occaDtype occaDtypeUlong4 = occa::c::newOccaType(occa::dtype::ulong4);

occaDtype occaDtypeLong2 = occa::c::newOccaType(occa::dtype::long2);
occaDtype occaDtypeLong3 = occa::c::newOccaType(occa::dtype::long3);
occaDtype occaDtypeLong4 = occa::c::newOccaType(occa::dtype::long4);

occaDtype occaDtypeFloat2 = occa::c::newOccaType(occa::dtype::float2);
occaDtype occaDtypeFloat3 = occa::c::newOccaType(occa::dtype::float3);
occaDtype occaDtypeFloat4 = occa::c::newOccaType(occa::dtype::float4);

occaDtype occaDtypeDouble2 = occa::c::newOccaType(occa::dtype::double2);
occaDtype occaDtypeDouble3 = occa::c::newOccaType(occa::dtype::double3);
occaDtype occaDtypeDouble4 = occa::c::newOccaType(occa::dtype::double4);
//======================================

OCCA_END_EXTERN_C
