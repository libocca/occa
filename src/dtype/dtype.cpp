#include <occa/defines.hpp>
#include <occa/dtype/builtins.hpp>
#include <occa/dtype/dtype.hpp>
#include <occa/tools/sys.hpp>

namespace occa {
  //---[ Dtype_T ]------------------------
  dtype_t::dtype_t() :
    ref(NULL),
    name_(),
    bytes_(0),
    global(false),
    tuple_(NULL),
    struct_(NULL) {}

  dtype_t::dtype_t(const std::string &name__,
                   const int bytes__,
                   const bool global_) :
    ref(NULL),
    name_(name__),
    bytes_(bytes__),
    global(global_),
    tuple_(NULL),
    struct_(NULL) {}

  dtype_t::dtype_t(const std::string &name__,
                   const dtype_t &other,
                   const bool global_) :
    ref(NULL),
    name_(),
    bytes_(0),
    global(false),
    tuple_(NULL),
    struct_(NULL) {

    *this = other;

    name_ = name__;
    global = global_;
  }

  dtype_t::dtype_t(const dtype_t &other) :
    ref(NULL),
    name_(),
    bytes_(0),
    global(false),
    tuple_(NULL),
    struct_(NULL) {

    *this = other;
  }

  dtype_t& dtype_t::operator = (const dtype_t &other_) {
    OCCA_ERROR("Cannot override global dtypes",
               !global);

    const dtype_t &other = other_.self();

    if (!ref || ref != &other) {
      delete tuple_;
      delete struct_;

      if (other.global) {
        // Clear values
        ref     = &other;
        name_   = "";
        bytes_  = 0;
        tuple_  = NULL;
        struct_ = NULL;
      } else {
        ref     = NULL;
        name_   = other.name_;
        bytes_  = other.bytes_;
        tuple_  = other.tuple_ ? other.tuple_->clone() : NULL;
        struct_ = other.struct_ ? other.struct_->clone() : NULL;
      }
    }
    return *this;
  }

  dtype_t::~dtype_t() {
    delete tuple_;
    delete struct_;
  }

  const std::string& dtype_t::name() const {
    return self().name_;
  }

  int dtype_t::bytes() const {
    return self().bytes_;
  }

  void dtype_t::setAsGlobal() {
    OCCA_ERROR("Unable to declare dtype references as global",
               ref == NULL);
    global = true;
  }

  bool dtype_t::isGlobal() const {
    return self().global;
  }

  // Tuple methods
  bool dtype_t::isTuple() const {
    return self().tuple_;
  }

  int dtype_t::tupleSize() const {
    const dtypeTuple_t *tuplePtr = self().tuple_;
    if (tuplePtr) {
      return tuplePtr->size;
    }
    return 0;
  }

  // Struct methods
  bool dtype_t::isStruct() const {
    return self().struct_;
  }

  int dtype_t::structFieldCount() const {
    const dtypeStruct_t *structPtr = self().struct_;
    if (structPtr) {
      return structPtr->fieldCount();
    }
    return 0;
  }

  const strVector& dtype_t::structFields() const {
    const dtypeStruct_t *structPtr = self().struct_;
    OCCA_ERROR("Cannot get fields from a non-struct dtype_t",
               structPtr != NULL);
    return structPtr->fieldNames;
  }

  const dtype_t& dtype_t::operator [] (const int field) const {
    const dtypeStruct_t *structPtr = self().struct_;
    OCCA_ERROR("Cannot access fields from a non-struct dtype_t",
               structPtr != NULL);
    return (*structPtr)[field];
  }

  const dtype_t& dtype_t::operator [] (const std::string &field) const {
    const dtypeStruct_t *structPtr = self().struct_;
    OCCA_ERROR("Cannot access fields from a non-struct dtype_t",
               structPtr != NULL);
    return (*structPtr)[field];
  }

  dtype_t& dtype_t::addField(const std::string &field,
                             const dtype_t &dtype,
                             const int tupleSize_) {
    OCCA_ERROR("Cannot add a field to a dtype_t reference",
               ref == NULL);
    OCCA_ERROR("Cannot add a field to an tuple dtype_t",
               tuple);
    OCCA_ERROR("Tuple size must be a positive integer",
               tupleSize_ > 0);

    if (!struct_) {
      struct_ = new dtypeStruct_t();
    }

    bytes_ += (dtype.bytes_ * tupleSize_);

    if (tupleSize_ == 1) {
      struct_->addField(field, dtype);
    } else {
      struct_->addField(field,
                        tuple(dtype, tupleSize_));
    }

    return *this;
  }

  void dtype_t::setFlattenedDtype() const {
    const dtype_t &self_ = self();
    if (!self_.flatDtype.size()) {
      self_.addFlatDtypes(flatDtype);
    }
  }

  void dtype_t::addFlatDtypes(dtypeVector_t &vec) const {
    const dtype_t &self_ = self();
    if (self_.struct_) {
      self_.struct_->addFlatDtypes(vec);
    } else if (self_.tuple_) {
      self_.tuple_->addFlatDtypes(vec);
    } else {
      vec.push_back(&self_);
    }
  }

  bool dtype_t::operator == (const dtype_t &other) const {
    return &(self()) == &(other.self());
  }

  bool dtype_t::operator != (const dtype_t &other) const {
    return !(*this == other);
  }

  bool dtype_t::matches(const dtype_t &other) const {
    const dtype_t &a = self();
    const dtype_t &b = other.self();

    // Check refs first
    if (&a == &b) {
      return true;
    }
    if (a.global != b.global) {
      return false;
    }
    // Refs didn't match and both a and b are globals
    if (a.global) {
        return false;
    }

    // Check type differences
    if (((bool) a.tuple_ != (bool) b.tuple_) ||
        ((bool) a.struct_ != (bool) b.struct_)) {
        return false;
    }
    // Check from the dtype type
    if (a.tuple_) {
      return a.tuple_->matches(*(b.tuple_));
    }
    if (a.struct_) {
      return a.struct_->matches(*(b.struct_));
    }

    // Shouldn't get here
    return false;
  }

  bool dtype_t::canBeCastedTo(const dtype_t &other) const {
    const dtype_t &from = self();
    const dtype_t &to   = other.self();

    // Anything can be casted from/to bytes
    if ((&from == &dtype::byte) ||
        (&to == &dtype::byte)) {
      return true;
    }

    from.setFlattenedDtype();
    to.setFlattenedDtype();

    const dtypeVector_t &fromVec = from.flatDtype;
    const int fromEntries = (int) fromVec.size();

    const dtypeVector_t &toVec = to.flatDtype;
    const int toEntries = (int) toVec.size();

    int entries = fromEntries;
    // Check if type cycles (e.g. float -> float2)
    if (fromEntries < toEntries) {
      if (!isCyclic(toVec, fromEntries)) {
        return false;
      }
      entries = fromEntries;
    } else if (fromEntries > toEntries) {
      if (!isCyclic(fromVec, toEntries)) {
        return false;
      }
      entries = toEntries;
    }

    for (int i = 0; i < entries; ++i) {
      if (fromVec[i] != toVec[i]) {
        return false;
      }
    }

    return true;
  }

  bool dtype_t::isCyclic(const dtypeVector_t &vec,
                         const int cycleLength) {
    const int size = (int) vec.size();
    if ((size % cycleLength) != 0) {
      return false;
    }

    const int cycles = size / cycleLength;
    for (int i = 0; i < cycleLength; ++i) {
      const dtype_t &dtype = *(vec[i]);
      for (int c = 1; c < cycles; ++c) {
        const dtype_t &dtype2 = *(vec[i + (c * cycleLength)]);
        if (dtype != dtype2) {
          return false;
        }
      }
    }

    return true;
  }

  json dtype_t::toJson() const {
    if (ref) {
      return ref->toJson();
    }

    if (tuple_) {
      return tuple_->toJson();
    } else if (struct_) {
      return struct_->toJson();
    }

    const dtype_t &dtype = dtype_t::getBuiltin(name_);
    if (&dtype != &dtype::none) {
      // "float"
      return name_;
    }

    // Custom type
    //   { name: "foo", bytes: 123 }
    json j;
    j["name"] = name_;
    j["bytes"] = bytes_;
    return j;
  }

  dtype_t dtype_t::tuple(const dtype_t &dtype,
                         const int size,
                         const bool global_) {
    dtype_t newType;
    newType.bytes_ = dtype.bytes_ * size;
    newType.tuple_ = new dtypeTuple_t(dtype, size);
    newType.global = global_;
    return newType;
  }

  const dtype_t& dtype_t::getBuiltin(const std::string name) {
    static dtypeGlobalMap_t dtypeMap;
    if (!dtypeMap.size()) {
      dtypeMap["none"] = &dtype::none;

      dtypeMap["void"] = &dtype::void_;
      dtypeMap["byte"] = &dtype::byte;

      dtypeMap["bool"]   = &dtype::bool_;
      dtypeMap["char"]   = &dtype::char_;
      dtypeMap["short"]  = &dtype::short_;
      dtypeMap["int"]    = &dtype::int_;
      dtypeMap["long"]   = &dtype::long_;
      dtypeMap["float"]  = &dtype::float_;
      dtypeMap["double"] = &dtype::double_;

      // Sized primitives
      dtypeMap["int8"]   = dtype::get<int8_t>().ref;
      dtypeMap["uint8"]  = dtype::get<uint8_t>().ref;
      dtypeMap["int16"]  = dtype::get<int16_t>().ref;
      dtypeMap["uint16"] = dtype::get<uint16_t>().ref;
      dtypeMap["int32"]  = dtype::get<int32_t>().ref;
      dtypeMap["uint32"] = dtype::get<uint32_t>().ref;
      dtypeMap["int64"]  = dtype::get<int64_t>().ref;
      dtypeMap["uint64"] = dtype::get<uint64_t>().ref;

      // OKL Primitives
      dtypeMap["uchar2"] = &dtype::uchar2;
      dtypeMap["uchar3"] = &dtype::uchar3;
      dtypeMap["uchar4"] = &dtype::uchar4;

      dtypeMap["char2"] = &dtype::char2;
      dtypeMap["char3"] = &dtype::char3;
      dtypeMap["char4"] = &dtype::char4;

      dtypeMap["ushort2"] = &dtype::ushort2;
      dtypeMap["ushort3"] = &dtype::ushort3;
      dtypeMap["ushort4"] = &dtype::ushort4;

      dtypeMap["short2"] = &dtype::short2;
      dtypeMap["short3"] = &dtype::short3;
      dtypeMap["short4"] = &dtype::short4;

      dtypeMap["uint2"] = &dtype::uint2;
      dtypeMap["uint3"] = &dtype::uint3;
      dtypeMap["uint4"] = &dtype::uint4;

      dtypeMap["int2"] = &dtype::int2;
      dtypeMap["int3"] = &dtype::int3;
      dtypeMap["int4"] = &dtype::int4;

      dtypeMap["ulong2"] = &dtype::ulong2;
      dtypeMap["ulong3"] = &dtype::ulong3;
      dtypeMap["ulong4"] = &dtype::ulong4;

      dtypeMap["long2"] = &dtype::long2;
      dtypeMap["long3"] = &dtype::long3;
      dtypeMap["long4"] = &dtype::long4;

      dtypeMap["float2"] = &dtype::float2;
      dtypeMap["float3"] = &dtype::float3;
      dtypeMap["float4"] = &dtype::float4;

      dtypeMap["double2"] = &dtype::double2;
      dtypeMap["double3"] = &dtype::double3;
      dtypeMap["double4"] = &dtype::double4;
    }
    dtypeGlobalMap_t::iterator it = dtypeMap.find(name);
    if (it != dtypeMap.end()) {
      return *(it->second);
    }

    return dtype::none;
  }

  dtype_t dtype_t::fromJson(const std::string &str) {
    json j;
    j.load(str);
    if (j.isInitialized()) {
      return dtype_t::fromJson(j);
    }
    return dtype::none;
  }

  dtype_t dtype_t::fromJson(const json &j) {
    if (j.isString()) {
      const std::string name = (std::string) j;
      const dtype_t &dtype = dtype_t::getBuiltin(name);
      OCCA_ERROR("Unknown builtin dtype [" << name << "]",
                 &dtype != &dtype::none);
      return dtype;
    }

    if (j.isObject()) {
      std::string name = j.get<std::string>("name");
      int bytes = j.get<int>("bytes");
      return dtype_t(name, bytes);
    }

    OCCA_ERROR("Incorrect dtype JSON format",
               j.isArray());

    const jsonArray &array = j.array();
    if ((array.size() == 2) && array[1].isNumber()) {
      return tuple(fromJson(array[0]),
                   (int) array[1]);
    }

    dtype_t dtype;
    const int fields = (int) array.size();
    for (int i = 0; i < fields; ++i) {
      const json &field = array[i];
      OCCA_ERROR("Incorrect dtype JSON format",
                 field.isArray() && (field.size() == 2) && field[1].isString());
      dtype.addField((std::string) field[0],
                     fromJson(field[1]));
    }
    return dtype;
  }

  std::ostream& operator << (std::ostream &out,
                             const dtype_t &dtype) {
    out << dtype.self().name_;
    return out;
  }
  //====================================


  //---[ Tuple ]----------------------
  dtypeTuple_t::dtypeTuple_t(const dtype_t &dtype_,
                             const int size_) :
    dtype(dtype_),
    size(size_) {}

  dtypeTuple_t* dtypeTuple_t::clone() const {
    return new dtypeTuple_t(dtype, size);
  }

  bool dtypeTuple_t::matches(const dtypeTuple_t &other) const {
    if (size != other.size) {
      return false;
    }
    return dtype.matches(other.dtype);
  }

  void dtypeTuple_t::addFlatDtypes(dtypeVector_t &vec) const {
    for (int i = 0; i < size; ++i) {
      dtype.addFlatDtypes(vec);
    }
  }

  json dtypeTuple_t::toJson() const {
    // Example:
    //   ['double', 2]
    json j;
    j.asArray();
    j += dtype.toJson();
    j += size;
    return j;
  }
  //====================================


  //---[ Struct ]-----------------------
  dtypeStruct_t::dtypeStruct_t() {}

  dtypeStruct_t* dtypeStruct_t::clone() const {
    dtypeStruct_t *s = new dtypeStruct_t();
    s->fieldNames = fieldNames;
    s->fieldTypes = fieldTypes;
    return s;
  }

  bool dtypeStruct_t::matches(const dtypeStruct_t &other) const {
    const int entries = (int) fieldNames.size();
    if (entries != (int) other.fieldNames.size()) {
      return false;
    }

    // Compare fields
    const std::string *names1 = &(fieldNames[0]);
    const std::string *names2 = &(other.fieldNames[0]);
    for (int i = 0; i < entries; ++i) {
      const std::string &name1 = names1[i];
      const std::string &name2 = names2[i];
      if (name1 != name2) {
        return false;
      }
      const dtype_t &dtype1 = fieldTypes.find(name1)->second;
      const dtype_t &dtype2 = fieldTypes.find(name2)->second;
      if (!dtype1.matches(dtype2)) {
        return false;
      }
    }

    return true;
  }

  int dtypeStruct_t::fieldCount() const {
    return (int) fieldNames.size();
  }

  const dtype_t& dtypeStruct_t::operator [] (const int field) const {
    OCCA_ERROR("Field index is out of bounds",
               (0 <= field) && (field < (int) fieldNames.size()));
    dtypeNameMap_t::const_iterator it = fieldTypes.find(fieldNames[field]);
    return it->second;
  }

  const dtype_t& dtypeStruct_t::operator [] (const std::string &field) const {
    dtypeNameMap_t::const_iterator it = fieldTypes.find(field);
    OCCA_ERROR("Field [" << field << "] is not in dtype_t",
               it != fieldTypes.end());
    return it->second;
  }

  void dtypeStruct_t::addField(const std::string &field,
                               const dtype_t &dtype) {
    const bool fieldExists = (fieldTypes.find(field) != fieldTypes.end());
    OCCA_ERROR("Field [" << field << "] is already in dtype_t",
               !fieldExists);

    if (!fieldExists) {
      fieldNames.push_back(field);
      fieldTypes[field] = dtype;
    }
  }

  void dtypeStruct_t::addFlatDtypes(dtypeVector_t &vec) const {
    const int entries = (int) fieldNames.size();
    const std::string *names = &(fieldNames[0]);
    for (int i = 0; i < entries; ++i) {
      const std::string &name = names[i];
      const dtype_t &dtype = fieldTypes.find(name)->second;
      dtype.addFlatDtypes(vec);
    }
  }

  json dtypeStruct_t::toJson() const {
    // Example:
    //   [['x', 'double'], ['y', 'double']]
    json j;
    j.asArray();

    const int entries = (int) fieldNames.size();
    const std::string *names = &(fieldNames[0]);
    for (int i = 0; i < entries; ++i) {
      const std::string &name = names[i];
      const dtype_t &dtype = fieldTypes.find(name)->second;

      json field;
      field.asArray();
      field += name;
      field += dtype.toJson();
      j += field;
    }

    return j;
  }
  //====================================
}
