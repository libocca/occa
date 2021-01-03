#include <cstring>

#include <occa/internal/c/types.hpp>
#include <occa/c/json.h>

OCCA_START_EXTERN_C

occaType occaCreateJson() {
  return occa::c::newOccaType(*(new occa::json()),
                              true);
}


//---[ Global methods ]-----------------
occaJson occaJsonParse(const char *c) {
  return occa::c::newOccaType(
    *(new occa::json(occa::json::parse(c))),
    true
  );
}

occaJson occaJsonRead(const char *filename) {
  return occa::c::newOccaType(
    *(new occa::json(occa::json::read(filename))),
    true
  );
}

void occaJsonWrite(occaJson j,
                   const char *filename) {
  occa::json &j_ = occa::c::json(j);
  j_.write(filename);
}

const char* occaJsonDump(occaJson j,
                         const int indent) {
  occa::json &j_ = occa::c::json(j);
  std::string str = j_.dump(indent);

  const size_t chars = str.size() + 1;
  char *c = (char*) ::malloc(chars);
  ::memcpy(c, str.c_str(), chars);

  return c;
}
//======================================


//---[ Type checks ]--------------------
bool occaJsonIsBoolean(occaJson j) {
  occa::json &j_ = occa::c::json(j);
  return j_.isBool();
}

bool occaJsonIsNumber(occaJson j) {
  occa::json &j_ = occa::c::json(j);
  return j_.isNumber();
}

bool occaJsonIsString(occaJson j) {
  occa::json &j_ = occa::c::json(j);
  return j_.isString();
}

bool occaJsonIsArray(occaJson j) {
  occa::json &j_ = occa::c::json(j);
  return j_.isArray();
}

bool occaJsonIsObject(occaJson j) {
  occa::json &j_ = occa::c::json(j);
  return j_.isObject();
}
//======================================


//---[ Casters ]------------------------
void occaJsonCastToBoolean(occaJson j) {
  occa::c::json(j).asBoolean();
}

void occaJsonCastToNumber(occaJson j) {
  occa::c::json(j).asNumber();
}

void occaJsonCastToString(occaJson j) {
  occa::c::json(j).asString();
}

void occaJsonCastToArray(occaJson j) {
  occa::c::json(j).asArray();
}

void occaJsonCastToObject(occaJson j) {
  occa::c::json(j).asObject();
}
//======================================


//---[ Getters ]------------------------
bool occaJsonGetBoolean(occaJson j) {
  occa::json &j_ = occa::c::json(j);
  return j_.boolean();
}

occaType occaJsonGetNumber(occaJson j,
                           const int type) {
  occa::json &j_ = occa::c::json(j);
  return occa::c::newOccaType(j_.number(), type);
}

const char* occaJsonGetString(occaJson j) {
  occa::json &j_ = occa::c::json(j);
  return j_.string().c_str();
}
//======================================


//---[ Object methods ]-----------------
occaType occaJsonObjectGet(occaJson j,
                           const char *key,
                           occaType defaultValue) {
  occa::json &j_ = occa::c::json(j);
  if (!j_.isInitialized()) {
    j_.asObject();
  }
  OCCA_ERROR("Input is not a JSON object",
             j_.isObject());

  if (j_.has(key)) {
    return occa::c::newOccaType(j_[key], false);
  }
  return defaultValue;
}

void occaJsonObjectSet(occaJson j,
                       const char *key,
                       occaType value) {
  occa::json &j_ = occa::c::json(j);
  if (!j_.isInitialized()) {
    j_.asObject();
  }
  OCCA_ERROR("Input is not a JSON object",
             j_.isObject());

  j_[key] = occa::c::inferJson(value);
}

bool occaJsonObjectHas(occaJson j,
                       const char *key) {
  occa::json &j_ = occa::c::json(j);
  if (!j_.isInitialized()) {
    j_.asObject();
  }
  OCCA_ERROR("Input is not a JSON object",
             j_.isObject());

  return j_.has(key);
}
//======================================


//---[ Array methods ]------------------
int occaJsonArraySize(occaJson j) {
  occa::json &j_ = occa::c::json(j);
  if (!j_.isInitialized()) {
    j_.asArray();
  }
  OCCA_ERROR("Input is not a JSON array",
             j_.isArray());

  return j_.size();
}

occaType occaJsonArrayGet(occaJson j,
                          const int index) {
  occa::json &j_ = occa::c::json(j);
  if (!j_.isInitialized()) {
    j_.asArray();
  }
  OCCA_ERROR("Input is not a JSON array",
             j_.isArray());

  return occa::c::newOccaType(j_[index], false);
}

void occaJsonArrayPush(occaJson j,
                       occaType value) {
  occa::json &j_ = occa::c::json(j);
  if (!j_.isInitialized()) {
    j_.asArray();
  }
  OCCA_ERROR("Input is not a JSON array",
             j_.isArray());

  j_ += occa::c::inferJson(value);
}

void occaJsonArrayPop(occaJson j) {
  occa::json &j_ = occa::c::json(j);
  if (!j_.isInitialized()) {
    j_.asArray();
  }
  OCCA_ERROR("Input is not a JSON array",
             j_.isArray());

  j_.array().pop_back();
}

void occaJsonArrayInsert(occaJson j,
                         const int index,
                         occaType value) {
  occa::json &j_ = occa::c::json(j);
  if (!j_.isInitialized()) {
    j_.asArray();
  }
  OCCA_ERROR("Input is not a JSON array",
             j_.isArray());

  occa::jsonArray &array = j_.array();
  OCCA_ERROR("Index [" << index << "] is out of bounds [0, "
             << array.size() << ')',
             (index >= 0) && (index < (int) array.size()));

  array.insert(array.begin() + index,
               occa::c::inferJson(value));
}

void occaJsonArrayClear(occaJson j) {
  occa::json &j_ = occa::c::json(j);
  if (!j_.isInitialized()) {
    j_.asArray();
  }
  OCCA_ERROR("Input is not a JSON array",
             j_.isArray());

  j_.array().clear();
}
//======================================

OCCA_END_EXTERN_C
