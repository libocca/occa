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
#include <occa/c/properties.h>

OCCA_START_EXTERN_C

occaType OCCA_RFUNC occaCreateProperties() {
  return occa::c::newOccaType(*(new occa::properties()));
}

occaType OCCA_RFUNC occaCreatePropertiesFromString(const char *c) {
  return occa::c::newOccaType(*(new occa::properties(c)));
}

void OCCA_RFUNC occaPropertiesSet(occaProperties props,
                                  const char *key,
                                  occaType value) {
  occa::properties& props_ = occa::c::properties(props);

  switch (value.type) {
  case occa::c::typeType::bool_:
    props_[key] = (bool) value.value.int8_; break;
  case occa::c::typeType::int8_:
    props_[key] = value.value.int8_; break;
  case occa::c::typeType::uint8_:
    props_[key] = value.value.uint8_; break;
  case occa::c::typeType::int16_:
    props_[key] = value.value.int16_; break;
  case occa::c::typeType::uint16_:
    props_[key] = value.value.uint16_; break;
  case occa::c::typeType::int32_:
    props_[key] = value.value.int32_; break;
  case occa::c::typeType::uint32_:
    props_[key] = value.value.uint32_; break;
  case occa::c::typeType::int64_:
    props_[key] = value.value.int64_; break;
  case occa::c::typeType::uint64_:
    props_[key] = value.value.uint64_; break;
  case occa::c::typeType::float_:
    props_[key] = value.value.float_; break;
  case occa::c::typeType::double_:
    props_[key] = value.value.double_; break;
  case occa::c::typeType::string:
    props_[key] = (char*) value.value.ptr; break;
  case occa::c::typeType::ptr:
    OCCA_ERROR("Invalid value type",
               value.value.ptr == NULL);
    props_[key].asNull(); break;
  default:
    OCCA_FORCE_ERROR("Invalid value type");
  }
}

occaType OCCA_RFUNC occaPropertiesGet(occaProperties props,
                                      const char *key,
                                      occaType defaultValue) {
  occa::properties& props_ = occa::c::properties(props);
  if (!props_.has(key)) {
    return defaultValue;
  }
  occa::json &value = props_[key];

  if (value.isString()) {
    return occaString(value.string().c_str());
  }
  if (value.isNumber()) {
    occa::primitive &number = value.number();
    switch(number.type) {
    case occa::primitiveType::int8_   : return occaInt8((int8_t) number);
    case occa::primitiveType::uint8_  : return occaUInt8((uint8_t) number);
    case occa::primitiveType::int16_  : return occaInt16((int16_t) number);
    case occa::primitiveType::uint16_ : return occaUInt16((uint16_t) number);
    case occa::primitiveType::int32_  : return occaInt32((int32_t) number);
    case occa::primitiveType::uint32_ : return occaUInt32((uint32_t) number);
    case occa::primitiveType::int64_  : return occaInt64((int64_t) number);
    case occa::primitiveType::uint64_ : return occaUInt64((uint64_t) number);
    case occa::primitiveType::float_  : return occaFloat((float) number);
    case occa::primitiveType::double_ : return occaDouble((double) number);
    }
  }
  if (value.isBoolean()) {
    return occaBool(value.boolean());
  }
  if (value.isObject() ||
      value.isArray()) {
    return occa::c::newOccaType((occa::properties&) value);
  }
  // Last type is NULL
  return occaNull;
}

OCCA_END_EXTERN_C
