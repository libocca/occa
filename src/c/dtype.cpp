#include <occa/c/types.hpp>
#include <occa/c/dtype.h>

OCCA_START_EXTERN_C

//-----[ Methods ]----------------------
OCCA_LFUNC occaDtype OCCA_RFUNC occaCreateDtype(const char *name,
                                                const int bytes) {
  return occa::c::newOccaType(
    *(new occa::dtype(name, bytes))
  );
}

OCCA_LFUNC const char* OCCA_RFUNC occaDtypeGetName(occaDtype type) {
  return occa::c::dtype(type).getName().c_str();
}

OCCA_LFUNC int OCCA_RFUNC occaDtypeGetBytes(occaDtype type) {
  return occa::c::dtype(type).getBytes();
}

OCCA_LFUNC void OCCA_RFUNC occaDtypeAddField(occaDtype type,
                                             const char *field,
                                             occaDtype fieldType) {
  occa::dtype &type_ = occa::c::dtype(type);
  type_.addField(field,
                 occa::c::dtype(fieldType));
}

OCCA_LFUNC int OCCA_RFUNC occaDtypeIsEqual(occaDtype a,
                                           occaDtype b) {
  return (occa::c::dtype(a) == occa::c::dtype(b));
}

OCCA_LFUNC occaDtype OCCA_RFUNC occaDtypeFromJson(occaJson json) {
  return occa::c::newOccaType(
    *(new occa::dtype(occa::dtype::fromJson(occa::c::json(json))))
  );
}
OCCA_LFUNC occaDtype OCCA_RFUNC occaDtypeFromJsonString(const char *str) {
  return occa::c::newOccaType(
    *(new occa::dtype(occa::dtype::fromJson(std::string(str))))
  );
}

OCCA_LFUNC occaJson OCCA_RFUNC occaDtypeToJson(occaDtype type) {
  occa::dtype &type_ = occa::c::dtype(type);
  return occa::c::newOccaType(
    *(new occa::json(type_.toJson())),
    true
  );
}
//======================================

//-----[ Builtins ]---------------------
occaDtype occaDtypeNone = occa::c::newOccaType(occa::dtypes::none);

occaDtype occaDtypeVoid = occa::c::newOccaType(occa::dtypes::void_);
occaDtype occaDtypeByte = occa::c::newOccaType(occa::dtypes::byte);

occaDtype occaDtypeBool   = occa::c::newOccaType(occa::dtypes::bool_);
occaDtype occaDtypeChar   = occa::c::newOccaType(occa::dtypes::char_);
occaDtype occaDtypeShort  = occa::c::newOccaType(occa::dtypes::short_);
occaDtype occaDtypeInt    = occa::c::newOccaType(occa::dtypes::int_);
occaDtype occaDtypeLong   = occa::c::newOccaType(occa::dtypes::long_);
occaDtype occaDtypeFloat  = occa::c::newOccaType(occa::dtypes::float_);
occaDtype occaDtypeDouble = occa::c::newOccaType(occa::dtypes::double_);

occaDtype occaDtypeInt8    = occa::c::newOccaType(occa::dtypes::int8);
occaDtype occaDtypeUint8   = occa::c::newOccaType(occa::dtypes::uint8);
occaDtype occaDtypeInt16   = occa::c::newOccaType(occa::dtypes::int16);
occaDtype occaDtypeUint16  = occa::c::newOccaType(occa::dtypes::uint16);
occaDtype occaDtypeInt32   = occa::c::newOccaType(occa::dtypes::int32);
occaDtype occaDtypeUint32  = occa::c::newOccaType(occa::dtypes::uint32);
occaDtype occaDtypeInt64   = occa::c::newOccaType(occa::dtypes::int64);
occaDtype occaDtypeUint64  = occa::c::newOccaType(occa::dtypes::uint64);
occaDtype occaDtypeFloat32 = occa::c::newOccaType(occa::dtypes::float32);
occaDtype occaDtypeFloat64 = occa::c::newOccaType(occa::dtypes::float64);
//======================================

OCCA_END_EXTERN_C
