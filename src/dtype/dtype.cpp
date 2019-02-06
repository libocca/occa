#include <occa/defines.hpp>
#include <occa/dtype/builtins.hpp>
#include <occa/dtype/dtype.hpp>
#include <occa/tools/sys.hpp>

namespace occa {
  dtype::dtype(const std::string &name_) :
    name(name_),
    bytes(0) {}

  dtype::dtype(const std::string &name_,
               const int bytes_) :
    name(name_),
    bytes(bytes_) {}

  dtype::dtype(const dtype &other) :
    name(other.name),
    bytes(other.bytes),
    fields(other.fields) {}

  const std::string& dtype::getName() const {
    return name;
  }

  int dtype::getBytes() const {
    return bytes;
  }

  dtype& dtype::addField(const std::string field,
                         const dtype &type) {
    bytes += type.bytes;
    const int fieldCount = (int) fields.size();
    for (int i = 0; i < fieldCount; ++i) {
      dtypeField &iField = fields[i];
      OCCA_ERROR("Field [" << field << "] is already defined",
                 iField.name != field);
    }
    fields.push_back(dtypeField(field, type));
    return *this;
  }

  bool dtype::operator == (const dtype &other) const {
    if (this == &other) {
      return true;
    }
    if ((bytes != other.bytes) ||
        (name != other.name)) {
      return false;
    }
    const int fieldCount = (int) fields.size();
    const int otherFieldCount = (int) other.fields.size();
    if (fieldCount != otherFieldCount) {
      return false;
    }
    for (int i = 0; i < fieldCount; ++i) {
      const dtypeField &field = fields[i];
      const dtypeField &otherField = other.fields[i];
      if (field.name != otherField.name) {
        return false;
      }
      if (field.type != otherField.type) {
        return false;
      }
    }
    return true;
  }

  bool dtype::operator != (const dtype &other) const {
    return !(*this == other);
  }

  const dtype& dtype::byName(const std::string name) {
    static dtypeNameMap_t dtypeMap;
    if (!dtypeMap.size()) {
      dtypeMap["void"] = &dtypes::void_;

      dtypeMap["bool"] = &dtypes::bool_;
      dtypeMap["char"] = &dtypes::char_;
      dtypeMap["short"] = &dtypes::short_;
      dtypeMap["int"] = &dtypes::int_;
      dtypeMap["long"] = &dtypes::long_;
      dtypeMap["float"] = &dtypes::float_;
      dtypeMap["double"] = &dtypes::double_;

      dtypeMap["int8"] = &dtypes::int8;
      dtypeMap["uint8"] = &dtypes::uint8;
      dtypeMap["int16"] = &dtypes::int16;
      dtypeMap["uint16"] = &dtypes::uint16;
      dtypeMap["int32"] = &dtypes::int32;
      dtypeMap["uint32"] = &dtypes::uint32;
      dtypeMap["int64"] = &dtypes::int64;
      dtypeMap["uint64"] = &dtypes::uint64;
      dtypeMap["float32"] = &dtypes::float32;
      dtypeMap["float64"] = &dtypes::float64;
    }
    dtypeNameMap_t::iterator it = dtypeMap.find(name);
    if (it != dtypeMap.end()) {
      return *(it->second);
    }
    return dtypes::none;
  }

  dtype dtype::fromJson(const std::string &str) {
    json j;
    j.load(str);
    return dtype::fromJson(j);
  }

  dtype dtype::fromJson(const json &j) {
    if (j.isString()) {
      const std::string name = (std::string) j;
      const dtype &type = dtype::byName(name);
      OCCA_ERROR("Unknown dtype [" << name << "]",
                 &type != &dtypes::none);
      return type;
    }

    OCCA_ERROR("Incorrect dtype json format",
               j.isObject() && (j.size() == 1));

    jsonObject::const_iterator it = j.object().begin();
    dtype type(it->first);

    const json &fieldsJson = it->second;
    OCCA_ERROR("Incorrect dtype json format",
               fieldsJson.isArray());

    const jsonArray &fields = fieldsJson.array();
    const int fieldCount = (int) fields.size();
    for (int i = 0; i < fieldCount; ++i) {
      const json &field = fields[i];
      OCCA_ERROR("Incorrect dtype json format",
                 field.isObject() && (field.size() == 1));

      it = field.object().begin();
      type.addField(it->first,
                    fromJson(it->second));
    }

    return type;
  }

  json dtype::toJson() const {
    const int fieldCount = (int) fields.size();
    if (!fieldCount) {
      return name;
    }
    json fieldsJson;
    fieldsJson.asArray();
    for (int i = 0; i < fieldCount; ++i) {
      const dtypeField &iField = fields[i];
      json fieldJson;
      fieldJson[iField.name] = iField.type.toJson();
      fieldsJson += fieldJson;
    }
    json j;
    j[name] = fieldsJson;
    return j;
  }

  std::ostream& operator << (std::ostream &out,
                             const dtype &type) {
    out << type.name;
    return out;
  }

  dtypeField::dtypeField(const std::string &name_,
                         const dtype &type_) :
    name(name_),
    type(type_) {}

  dtypeField::dtypeField(const dtypeField &other) :
    name(other.name),
    type(other.type) {}
}
