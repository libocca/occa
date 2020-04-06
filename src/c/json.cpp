#include <occa/c/types.hpp>
#include <occa/c/json.h>

OCCA_START_EXTERN_C

occaType OCCA_RFUNC occaCreateJson() {
  return occa::c::newOccaType(*(new occa::json()),
                              true);
}


//---[ Global methods ]-----------------
occaJson OCCA_RFUNC occaJsonParse(const char *c) {
  return occa::c::newOccaType(
    *(new occa::json(occa::json::parse(c))),
    true
  );
}

OCCA_LFUNC occaJson OCCA_RFUNC occaJsonRead(const char *filename) {
  return occa::c::newOccaType(
    *(new occa::json(occa::json::read(filename))),
    true
  );
}

OCCA_LFUNC void OCCA_RFUNC occaJsonWrite(occaJson j,
                                         const char *filename) {
  occa::json &j_ = occa::c::json(j);
  j_.write(filename);
}

OCCA_LFUNC const char* OCCA_RFUNC occaJsonDump(occaJson j,
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
int OCCA_RFUNC occaJsonIsBoolean(occaJson j) {
  occa::json &j_ = occa::c::json(j);
  return j_.isBoolean();
}

int OCCA_RFUNC occaJsonIsNumber(occaJson j) {
  occa::json &j_ = occa::c::json(j);
  return j_.isNumber();
}

int OCCA_RFUNC occaJsonIsString(occaJson j) {
  occa::json &j_ = occa::c::json(j);
  return j_.isString();
}

int OCCA_RFUNC occaJsonIsArray(occaJson j) {
  occa::json &j_ = occa::c::json(j);
  return j_.isArray();
}

int OCCA_RFUNC occaJsonIsObject(occaJson j) {
  occa::json &j_ = occa::c::json(j);
  return j_.isObject();
}
//======================================


//---[ Casters ]------------------------
void OCCA_RFUNC occaJsonCastToBoolean(occaJson j) {
  occa::c::json(j).asBoolean();
}

void OCCA_RFUNC occaJsonCastToNumber(occaJson j) {
  occa::c::json(j).asNumber();
}

void OCCA_RFUNC occaJsonCastToString(occaJson j) {
  occa::c::json(j).asString();
}

void OCCA_RFUNC occaJsonCastToArray(occaJson j) {
  occa::c::json(j).asArray();
}

void OCCA_RFUNC occaJsonCastToObject(occaJson j) {
  occa::c::json(j).asObject();
}
//======================================


//---[ Getters ]------------------------
int OCCA_RFUNC occaJsonGetBoolean(occaJson j) {
  occa::json &j_ = occa::c::json(j);
  return (int) j_.boolean();
}

occaType OCCA_RFUNC occaJsonGetNumber(occaJson j,
                                      const int type) {
  occa::json &j_ = occa::c::json(j);
  return occa::c::newOccaType(j_.number(), type);
}

const char* OCCA_RFUNC occaJsonGetString(occaJson j) {
  occa::json &j_ = occa::c::json(j);
  return j_.string().c_str();
}
//======================================


//---[ Object methods ]-----------------
occaType OCCA_RFUNC occaJsonObjectGet(occaJson j,
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

void OCCA_RFUNC occaJsonObjectSet(occaJson j,
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

int OCCA_RFUNC occaJsonObjectHas(occaJson j,
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
int OCCA_RFUNC occaJsonArraySize(occaJson j) {
  occa::json &j_ = occa::c::json(j);
  if (!j_.isInitialized()) {
    j_.asArray();
  }
  OCCA_ERROR("Input is not a JSON array",
             j_.isArray());

  return j_.size();
}

occaType OCCA_RFUNC occaJsonArrayGet(occaJson j,
                                     const int index) {
  occa::json &j_ = occa::c::json(j);
  if (!j_.isInitialized()) {
    j_.asArray();
  }
  OCCA_ERROR("Input is not a JSON array",
             j_.isArray());

  return occa::c::newOccaType(j_[index], false);
}

void OCCA_RFUNC occaJsonArrayPush(occaJson j,
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

void OCCA_RFUNC occaJsonArrayInsert(occaJson j,
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

void OCCA_RFUNC occaJsonArrayClear(occaJson j) {
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
