#include <algorithm>

#include <occa/defines.hpp>
#include <occa/dtype/builtins.hpp>
#include <occa/dtype/dtype.hpp>
#include <occa/dtype/utils.hpp>
#include <occa/types/json.hpp>
#include <occa/internal/utils/sys.hpp>

namespace occa {
  //---[ Dtype_T ]------------------------
  dtype_t::dtype_t() :
    ref(NULL),
    name_(),
    bytes_(0),
    registered(false),
    enum_(NULL),
    struct_(NULL),
    tuple_(NULL),
    union_(NULL) {}

  dtype_t::dtype_t(const std::string &name__,
                   const int bytes__,
                   const bool registered_) :
    ref(NULL),
    name_(name__),
    bytes_(bytes__),
    registered(registered_),
    enum_(NULL),
    struct_(NULL),
    tuple_(NULL),
    union_(NULL) {}

  dtype_t::dtype_t(const std::string &name__,
                   const dtype_t &other,
                   const bool registered_) :
    ref(NULL),
    name_(),
    bytes_(0),
    registered(false),
    enum_(NULL),
    struct_(NULL),
    tuple_(NULL),
    union_(NULL) {

    *this = other;

    name_ = name__;
    registered = registered_;
  }

  dtype_t::dtype_t(const dtype_t &other) :
    ref(NULL),
    name_(),
    bytes_(0),
    registered(false),
    enum_(NULL),
    struct_(NULL),
    tuple_(NULL),
    union_(NULL) {

    *this = other;
  }

  dtype_t& dtype_t::operator = (const dtype_t &other_) {
    OCCA_ERROR("Cannot override registered dtypes",
               !registered);

    const dtype_t &other = other_.self();

    if (!ref || ref != &other) {
      delete enum_;
      delete struct_;
      delete tuple_;
      delete union_;

      if (other.registered) {
        // Clear values
        ref       = &other;
        name_     = "";
        bytes_    = 0;
        enum_     = NULL;
        struct_   = NULL;
        tuple_    = NULL;
        union_    = NULL;
      } else {
        ref       = NULL;
        name_     = other.name_;
        bytes_    = other.bytes_;
        enum_     = other.enum_ ? other.enum_->clone() : NULL;
        struct_   = other.struct_ ? other.struct_->clone() : NULL;
        tuple_    = other.tuple_ ? other.tuple_->clone() : NULL;
        union_    = other.union_ ? other.union_->clone() : NULL;
      }
    }
    return *this;
  }

  dtype_t::~dtype_t() {
    delete enum_;
    delete struct_;
    delete tuple_;
    delete union_;
  }

  const std::string& dtype_t::name() const {
    return self().name_;
  }

  int dtype_t::bytes() const {
    return self().bytes_;
  }

  void dtype_t::registerType() {
    OCCA_ERROR("Unable to register dtype references", ref == NULL);
    registered = true;
  }

  bool dtype_t::isRegistered() const {
    return self().registered;
  }

  // Enum methods
  bool dtype_t::isEnum() const {
    return self().enum_;
  }

  int dtype_t::enumEnumeratorCount() const {
    const dtypeEnum_t *enumPtr = self().enum_;
    if (enumPtr) {
      return enumPtr->enumeratorCount();
    }
    return 0;
  }

  const strVector& dtype_t::enumEnumeratorNames() const {
    const dtypeEnum_t *enumPtr = self().enum_;
    OCCA_ERROR("Cannot get enumerators from a non-enum dtype_t", enumPtr != NULL);
    return enumPtr->enumeratorNames;
  }

  dtype_t& dtype_t::addEnumerator(const std::string &enumerator) {

    if (!enum_) {
      enum_ = new dtypeEnum_t();
    }

    enum_->addEnumerator(enumerator);

    return *this;
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

  const strVector& dtype_t::structFieldNames() const {
    const dtypeStruct_t *structPtr = self().struct_;
    OCCA_ERROR("Cannot get fields from a non-struct dtype_t", structPtr != NULL);
    return structPtr->fieldNames;
  }

  const dtype_t& dtype_t::operator [] (const int field) const {
    if (self().union_) {
      const dtypeUnion_t *unionPtr = self().union_;
      OCCA_ERROR("Cannot access fields from a non-union dtype_t", unionPtr != NULL);
      return (*unionPtr)[field];
    } else {
      const dtypeStruct_t *structPtr = self().struct_;
      OCCA_ERROR("Cannot access fields from a non-struct dtype_t", structPtr != NULL);
      return (*structPtr)[field];
    }
  }

  const dtype_t& dtype_t::operator [] (const std::string &field) const {
    if (self().union_) {
      const dtypeUnion_t *unionPtr = self().union_;
      OCCA_ERROR("Cannot access fields from a non-union dtype_t", unionPtr != NULL);
      return (*unionPtr)[field];
    } else {
      const dtypeStruct_t *structPtr = self().struct_;
      OCCA_ERROR("Cannot access fields from a non-struct dtype_t", structPtr != NULL);
      return (*structPtr)[field];
    }
  }
  // Union methods
  bool dtype_t::isUnion() const {
    return self().union_;
  }

  int dtype_t::unionFieldCount() const {
    const dtypeUnion_t *unionPtr = self().union_;
    if (unionPtr) {
      return unionPtr->fieldCount();
    }
    return 0;
  }

  const strVector& dtype_t::unionFieldNames() const {
    const dtypeUnion_t *unionPtr = self().union_;
    OCCA_ERROR("Cannot get fields from a non-union dtype_t", unionPtr != NULL);
    return unionPtr->fieldNames;
  }

  dtype_t& dtype_t::addField(const std::string &field,
                             const dtype_t &dtype,
                             const int tupleSize_) {
    OCCA_ERROR("Cannot add a field to a dtype_t reference", ref == NULL);
    OCCA_ERROR("Cannot add a field to an tuple dtype_t", tuple_ == NULL);
    OCCA_ERROR("Tuple size must be a positive integer", tupleSize_ > 0);

    if (self().union_) {
      if (!union_) {
        union_ = new dtypeUnion_t();
      }

      bytes_ += (dtype.bytes_ * tupleSize_);

      if (tupleSize_ == 1) {
        union_->addField(field, dtype);
      } else {
        union_->addField(field, tuple(dtype, tupleSize_));
      }
    } else {
      if (!struct_) {
        struct_ = new dtypeStruct_t();
      }

      bytes_ += (dtype.bytes_ * tupleSize_);

      if (tupleSize_ == 1) {
        struct_->addField(field, dtype);
      } else {
        struct_->addField(field, tuple(dtype, tupleSize_));
      }
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
    } else if (self_.union_) {
      self_.union_->addFlatDtypes(vec);
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

  const dtype_t& dtype_t::operator || (const dtype_t &other) const {
    return (
      (*this == dtype::none)
      ? other
      : *this
    );
  }

  bool dtype_t::matches(const dtype_t &other) const {
    const dtype_t &a = self();
    const dtype_t &b = other.self();

    // Check refs first
    if (&a == &b) {
      return true;
    }
    if (a.registered != b.registered) {
      return false;
    }

    // Refs didn't match and both a and b are registered
    if (a.registered) {
        return false;
    }

    // Check type differences
    if (((bool) a.enum_ != (bool) b.enum_) ||
        ((bool) a.struct_ != (bool) b.struct_) ||
        ((bool) a.tuple_ != (bool) b.tuple_) ||
        ((bool) a.union_ != (bool) b.union_)) {
        return false;
    }
    // Check from the dtype type
    if (a.enum_) {
      return a.enum_->matches(*(b.enum_));
    }
    if (a.struct_) {
      return a.struct_->matches(*(b.struct_));
    }
    if (a.tuple_) {
      return a.tuple_->matches(*(b.tuple_));
    }
    if (a.union_) {
      return a.union_->matches(*(b.union_));
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

  dtype_t dtype_t::tuple(const dtype_t &dtype,
                         const int size,
                         const bool registered_) {
    dtype_t newType;
    newType.bytes_ = dtype.bytes_ * size;
    newType.tuple_ = new dtypeTuple_t(dtype, size);
    newType.registered = registered_;
    return newType;
  }

  const dtype_t& dtype_t::getBuiltin(const std::string &name) {
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
      dtypeMap["unsigned long"] = &dtype::ulong_;

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

  json dtype_t::toJson(const std::string &name) const {
    json output;
    toJson(output, name);
    return output;
  }

  void dtype_t::toJson(json &j, const std::string &name) const {
    if (ref) {
      return ref->toJson(j, name);
    }

    if (enum_) {
      return enum_->toJson(j, name);
    } else if (struct_) {
      return struct_->toJson(j, name);
    } else if (tuple_) {
      return tuple_->toJson(j, name);
    } else if (union_) {
      return union_->toJson(j, name);
    }

    j.clear();
    j.asObject();
    const dtype_t &dtype = dtype_t::getBuiltin(name_);
    if (&dtype != &dtype::none) {
      j["type"] = "builtin";
      j["name"] = name_;
    } else {
      j["type"]  = "custom";
      j["name"]  = name_;
      j["bytes"] = bytes_;
    }
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
    const std::string type = j["type"].toString();

    dtype_t dtype;
    dtype.name_ = j["name"].toString();

    if (type == "builtin") {
      const dtype_t &builtin = dtype_t::getBuiltin(dtype.name_);
      OCCA_ERROR("Unknown dtype builtin [" << dtype.name_ << "]",
                 &builtin != &dtype::none);
      dtype = builtin;
    } else if (type == "enum") {
      dtype.enum_ = dtypeEnum_t::fromJson(j).clone();
    } else if (type == "struct") {
      dtype.struct_ = dtypeStruct_t::fromJson(j).clone();
    } else if (type == "tuple") {
      dtype.tuple_ = dtypeTuple_t::fromJson(j).clone();
    } else if (type == "union") {
      dtype.union_ = dtypeUnion_t::fromJson(j).clone();
    } else if (type == "custom") {
      dtype.bytes_ = (int) j["bytes"];
    } else {
      OCCA_FORCE_ERROR("Incorrect dtype JSON format");
    }

    return dtype;
  }

  std::string dtype_t::toString(const std::string &varName) const {
    std::stringstream ss;
    const dtype_t &self_ = self();

    std::string name;
    if (varName.size()) {
      name = varName;
    } else {
      name = self_.name_;
    }

    if (self_.enum_) {
      ss << self_.enum_->toString(name);
    } else if (self_.struct_) {
      ss << self_.struct_->toString(name);
    } else if (self_.tuple_) {
      ss << self_.tuple_->toString(name);
    } else if (self_.union_) {
      ss << self_.union_->toString(name);
    } else {
      ss << name;
    }

    return ss.str();
  }

  std::ostream& operator << (std::ostream &out,
                             const dtype_t &dtype) {
    out << dtype.toString();
    return out;
  }
  //====================================


  //---[ Enum ]-----------------------
  dtypeEnum_t::dtypeEnum_t() {}

  dtypeEnum_t* dtypeEnum_t::clone() const {
    dtypeEnum_t *s = new dtypeEnum_t();
    s->enumeratorNames = enumeratorNames;
    return s;
  }

  bool dtypeEnum_t::matches(const dtypeEnum_t &other) const {
    const int enumeratorCount = (int) enumeratorNames.size();
    if (enumeratorCount != (int) other.enumeratorNames.size()) {
      return false;
    }

    // Compare enumerators
    const std::string *names1 = &(enumeratorNames[0]);
    const std::string *names2 = &(other.enumeratorNames[0]);
    for (int i = 0; i < enumeratorCount; ++i) {
      const std::string &name1 = names1[i];
      const std::string &name2 = names2[i];
      if (name1 != name2) {
        return false;
      }
    }

    return true;
  }

  int dtypeEnum_t::enumeratorCount() const {
    return (int) enumeratorNames.size();
  }

  void dtypeEnum_t::addEnumerator(const std::string &enumerator) {
    const bool enumeratorExists = std::find(enumeratorNames.begin(), enumeratorNames.end(), enumerator) != enumeratorNames.end();
    OCCA_ERROR("Enumerator [" << enumerator << "] is already in dtype_t", !enumeratorExists);

    if (!enumeratorExists) {
      enumeratorNames.push_back(enumerator);
    }
  }

  void dtypeEnum_t::toJson(json &j, const std::string &name) const {
    j.clear();
    j.asObject();

    j["type"] = "enum";
    if (name.size()) {
      j["name"] = name;
    }

    json &enumeratorsJson = j["enumerators"].asArray();
    const int enumeratorCount = (int) enumeratorNames.size();

    const std::string *names = &(enumeratorNames[0]);
    for (int i = 0; i < enumeratorCount; ++i) {
      const std::string &enumeratorName = names[i];

      json enumeratorJson;
      enumeratorJson["name"] = enumeratorName;
      enumeratorsJson += enumeratorJson;
    }
  }

  dtypeEnum_t dtypeEnum_t::fromJson(const json &j) {
    OCCA_ERROR("JSON enumerator [enumerators] missing from enum", j.has("enumerators"));
    OCCA_ERROR("JSON enumerator [enumerators] must be an array of dtypes", j["enumerators"].isArray());

    const jsonArray &enumerators = j["enumerators"].array();
    const int enumeratorCount = (int) enumerators.size();

    dtypeEnum_t enum_;
    for (int i = 0; i < enumeratorCount; ++i) {
      const json &enumeratorJson = enumerators[i];
      OCCA_ERROR("JSON enumerator [name] missing from enum enumerator", enumeratorJson.has("name"));
      OCCA_ERROR("JSON enumerator [name] must be a string for enum enumerators", enumeratorJson["name"].isString());

      enum_.addEnumerator(enumeratorJson["name"].string());
    }

    return enum_;
  }

  std::string dtypeEnum_t::toString(const std::string &enumName) const {
    std::stringstream ss;
    const int enumeratorCount = (int) enumeratorNames.size();

    ss << "enum ";
    if (enumName.size()) {
      ss << enumName << ' ';
    }
    ss << '{';

    if (!enumeratorCount) {
      ss << '}';
      return ss.str();
    }

    ss << '\n';

    const std::string *names = &(enumeratorNames[0]);
    dtype_t prevDtype = dtype::none;
    for (int i = 0; i < enumeratorCount; ++i) {
      const std::string &name = names[i];
      if (i) {
        ss << ", ";
      }
      ss << name;
    }
    ss << "\n}";

    return ss.str();
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
    const int fieldCount = (int) fieldNames.size();
    if (fieldCount != (int) other.fieldNames.size()) {
      return false;
    }

    // Compare fields
    const std::string *names1 = &(fieldNames[0]);
    const std::string *names2 = &(other.fieldNames[0]);
    for (int i = 0; i < fieldCount; ++i) {
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
    const int fieldCount = (int) fieldNames.size();
    const std::string *names = &(fieldNames[0]);
    for (int i = 0; i < fieldCount; ++i) {
      const std::string &name = names[i];
      const dtype_t &dtype = fieldTypes.find(name)->second;
      dtype.addFlatDtypes(vec);
    }
  }

  void dtypeStruct_t::toJson(json &j, const std::string &name) const {
    j.clear();
    j.asObject();

    j["type"] = "struct";
    if (name.size()) {
      j["name"] = name;
    }

    json &fieldsJson = j["fields"].asArray();
    const int fieldCount = (int) fieldNames.size();

    const std::string *names = &(fieldNames[0]);
    for (int i = 0; i < fieldCount; ++i) {
      const std::string &fieldName = names[i];
      const dtype_t &dtype = fieldTypes.find(fieldName)->second;

      json fieldJson;
      fieldJson["dtype"] = dtype::toJson(dtype);
      fieldJson["name"] = fieldName;
      fieldsJson += fieldJson;
    }
  }

  dtypeStruct_t dtypeStruct_t::fromJson(const json &j) {
    OCCA_ERROR("JSON field [fields] missing from struct",
               j.has("fields"));
    OCCA_ERROR("JSON field [fields] must be an array of dtypes",
               j["fields"].isArray());

    const jsonArray &fields = j["fields"].array();
    const int fieldCount = (int) fields.size();

    dtypeStruct_t struct_;
    for (int i = 0; i < fieldCount; ++i) {
      const json &fieldJson = fields[i];
      OCCA_ERROR("JSON field [dtype] missing from struct field",
                 fieldJson.has("dtype"));
      OCCA_ERROR("JSON field [name] missing from struct field",
                 fieldJson.has("name"));
      OCCA_ERROR("JSON field [name] must be a string for struct fields",
                 fieldJson["name"].isString());

      struct_.addField(fieldJson["name"].string(),
                       dtype_t::fromJson(fieldJson["dtype"]));
    }

    return struct_;
  }

  std::string dtypeStruct_t::toString(const std::string &varName) const {
    std::stringstream ss;
    const int fieldCount = (int) fieldNames.size();

    ss << "struct ";
    if (varName.size()) {
      ss << varName << ' ';
    }
    ss << '{';

    if (!fieldCount) {
      ss << '}';
      return ss.str();
    }

    ss << '\n';

    const std::string *names = &(fieldNames[0]);
    dtype_t prevDtype = dtype::none;
    for (int i = 0; i < fieldCount; ++i) {
      const std::string &name = names[i];
      const dtype_t &dtype = fieldTypes.find(name)->second;

      if (prevDtype != dtype) {
        prevDtype = dtype;
        if (i) {
          ss << ";\n";
        }
        ss << "  " << dtype.toString(name);
      } else {
        if (!i) {
          prevDtype = dtype;
        }
        ss << ", " << name;
      }
    }
    ss << ";\n}";

    return ss.str();
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

  void dtypeTuple_t::toJson(json &j, const std::string &name) const {
    j.clear();
    j.asObject();

    j["type"]  = "tuple";
    if (name.size()) {
      j["name"]  = name;
    }
    j["dtype"] = dtype::toJson(dtype);
    j["size"]  = size;
  }

  dtypeTuple_t dtypeTuple_t::fromJson(const json &j) {
    OCCA_ERROR("JSON field [dtype] missing from tuple",
               j.has("dtype"));
    OCCA_ERROR("JSON field [size] missing from tuple",
               j.has("size"));
    OCCA_ERROR("JSON field [size] must be an integer",
               j["size"].isNumber());

    return dtypeTuple_t(dtype_t::fromJson(j["dtype"]),
                        (int) j["size"]);
  }

  std::string dtypeTuple_t::toString(const std::string &varName) const {
    std::stringstream ss;

    ss << dtype;

    if (varName.size()) {
      ss << ' ' << varName;
    }

    ss << '[';
    if (size >= 0) {
      ss << size;
    } else {
      ss << '?';
    }
    ss << ']';

    return ss.str();
  }
  //====================================

  //---[ Union ]-----------------------
  dtypeUnion_t::dtypeUnion_t() {}

  dtypeUnion_t* dtypeUnion_t::clone() const {
    dtypeUnion_t *s = new dtypeUnion_t();
    s->fieldNames = fieldNames;
    s->fieldTypes = fieldTypes;
    return s;
  }

  bool dtypeUnion_t::matches(const dtypeUnion_t &other) const {
    const int fieldCount = (int) fieldNames.size();
    if (fieldCount != (int) other.fieldNames.size()) {
      return false;
    }

    // Compare fields
    const std::string *names1 = &(fieldNames[0]);
    const std::string *names2 = &(other.fieldNames[0]);
    for (int i = 0; i < fieldCount; ++i) {
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

  int dtypeUnion_t::fieldCount() const {
    return (int) fieldNames.size();
  }

  const dtype_t& dtypeUnion_t::operator [] (const int field) const {
    OCCA_ERROR("Field index is out of bounds",
               (0 <= field) && (field < (int) fieldNames.size()));
    dtypeNameMap_t::const_iterator it = fieldTypes.find(fieldNames[field]);
    return it->second;
  }

  const dtype_t& dtypeUnion_t::operator [] (const std::string &field) const {
    dtypeNameMap_t::const_iterator it = fieldTypes.find(field);
    OCCA_ERROR("Field [" << field << "] is not in dtype_t",
               it != fieldTypes.end());
    return it->second;
  }

  void dtypeUnion_t::addField(const std::string &field,
                               const dtype_t &dtype) {
    const bool fieldExists = (fieldTypes.find(field) != fieldTypes.end());
    OCCA_ERROR("Field [" << field << "] is already in dtype_t",
               !fieldExists);

    if (!fieldExists) {
      fieldNames.push_back(field);
      fieldTypes[field] = dtype;
    }
  }

  void dtypeUnion_t::addFlatDtypes(dtypeVector_t &vec) const {
    const int fieldCount = (int) fieldNames.size();
    const std::string *names = &(fieldNames[0]);
    for (int i = 0; i < fieldCount; ++i) {
      const std::string &name = names[i];
      const dtype_t &dtype = fieldTypes.find(name)->second;
      dtype.addFlatDtypes(vec);
    }
  }

  void dtypeUnion_t::toJson(json &j, const std::string &name) const {
    j.clear();
    j.asObject();

    j["type"] = "union";
    if (name.size()) {
      j["name"] = name;
    }

    json &fieldsJson = j["fields"].asArray();
    const int fieldCount = (int) fieldNames.size();

    const std::string *names = &(fieldNames[0]);
    for (int i = 0; i < fieldCount; ++i) {
      const std::string &fieldName = names[i];
      const dtype_t &dtype = fieldTypes.find(fieldName)->second;

      json fieldJson;
      fieldJson["dtype"] = dtype::toJson(dtype);
      fieldJson["name"] = fieldName;
      fieldsJson += fieldJson;
    }
  }

  dtypeUnion_t dtypeUnion_t::fromJson(const json &j) {
    OCCA_ERROR("JSON field [fields] missing from union",
               j.has("fields"));
    OCCA_ERROR("JSON field [fields] must be an array of dtypes",
               j["fields"].isArray());

    const jsonArray &fields = j["fields"].array();
    const int fieldCount = (int) fields.size();

    dtypeUnion_t union_;
    for (int i = 0; i < fieldCount; ++i) {
      const json &fieldJson = fields[i];
      OCCA_ERROR("JSON field [dtype] missing from union field",
                 fieldJson.has("dtype"));
      OCCA_ERROR("JSON field [name] missing from union field",
                 fieldJson.has("name"));
      OCCA_ERROR("JSON field [name] must be a string for union fields",
                 fieldJson["name"].isString());

      union_.addField(fieldJson["name"].string(),
                       dtype_t::fromJson(fieldJson["dtype"]));
    }

    return union_;
  }

  std::string dtypeUnion_t::toString(const std::string &varName) const {
    std::stringstream ss;
    const int fieldCount = (int) fieldNames.size();

    ss << "union ";
    if (varName.size()) {
      ss << varName << ' ';
    }
    ss << '{';

    if (!fieldCount) {
      ss << '}';
      return ss.str();
    }

    ss << '\n';

    const std::string *names = &(fieldNames[0]);
    dtype_t prevDtype = dtype::none;
    for (int i = 0; i < fieldCount; ++i) {
      const std::string &name = names[i];
      const dtype_t &dtype = fieldTypes.find(name)->second;

      if (prevDtype != dtype) {
        prevDtype = dtype;
        if (i) {
          ss << ";\n";
        }
        ss << "  " << dtype.toString(name);
      } else {
        if (!i) {
          prevDtype = dtype;
        }
        ss << ", " << name;
      }
    }
    ss << ";\n}";

    return ss.str();
  }
  //====================================



}
