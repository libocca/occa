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
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <occa.hpp>
#include <occa/c/properties.h>
#include <occa/c/types.hpp>
#include <occa/tools/testing.hpp>

void testProperties();

int main(const int argc, const char **argv) {
  srand(time(NULL));

  testProperties();
}
void testProperties() {
  occaProperties cProps = occaCreateProperties();
  occa::properties &props = occa::c::properties(cProps);

#define TEST_SET_PROP(propType, OCCA_TYPE, propValue, field, func)  \
  do {                                                                \
    propType v = (propType) (propValue);                              \
    occaPropertiesSet(cProps, #func, func(v));                        \
    occaType value_ = occaPropertiesGet(cProps, #func, occaDefault);  \
    ASSERT_EQ_BINARY(value_.type, OCCA_TYPE);                         \
    ASSERT_EQ((propType) value_.value.field, v);                      \
  } while (0)

  TEST_SET_PROP(bool, OCCA_BOOL, rand() % 2, int8_, occaBool);

  TEST_SET_PROP(int8_t, OCCA_INT8, rand(), int8_, occaInt8);
  TEST_SET_PROP(uint8_t, OCCA_UINT8, rand(), uint8_, occaUInt8);

  TEST_SET_PROP(int16_t, OCCA_INT16, rand(), int16_, occaInt16);
  TEST_SET_PROP(uint16_t, OCCA_UINT16, rand(), uint16_, occaUInt16);

  TEST_SET_PROP(int32_t, OCCA_INT32, rand(), int32_, occaInt32);
  TEST_SET_PROP(uint32_t, OCCA_UINT32, rand(), uint32_, occaUInt32);

  TEST_SET_PROP(int64_t, OCCA_INT64, rand(), int64_, occaInt64);
  TEST_SET_PROP(uint64_t, OCCA_UINT64, rand(), uint64_, occaUInt64);

  TEST_SET_PROP(float, OCCA_FLOAT, rand() / ((double) rand() + 1.0), float_, occaFloat);
  TEST_SET_PROP(double, OCCA_DOUBLE, rand() / ((double) rand() + 1.0), double_, occaDouble);

  const std::string stringValue = occa::toString(rand());
  TEST_SET_PROP(const char*, OCCA_STRING, stringValue.c_str(), ptr, occaString);

#undef TEST_SET_PROP

  const std::string propStr = (std::string) props;
  occaProperties cProps2 = occaCreatePropertiesFromString(propStr.c_str());
  occa::properties &props2 = occa::c::properties(cProps2);

  ASSERT_EQ(props,
            props2);

  occaFree(cProps);
  occaFree(cProps2);
}
