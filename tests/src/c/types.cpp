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
#include <occa/c/types.hpp>
#include <occa/tools/testing.hpp>

void testTypes();

int main(const int argc, const char **argv) {
  srand(time(NULL));

  testTypes();
}

void testTypes() {
#define TEST_OCCA_TYPE(value, OCCA_TYPE)        \
  do {                                          \
    occaType v = occa::c::newOccaType(value);   \
    ASSERT_EQ(v.type, OCCA_TYPE);               \
    occaFree(v);                                \
  } while (0)

  TEST_OCCA_TYPE((void*) NULL, OCCA_PTR);

  TEST_OCCA_TYPE(true, OCCA_BOOL);
  TEST_OCCA_TYPE((int8_t) 1, OCCA_INT8);
  TEST_OCCA_TYPE((uint8_t) 1, OCCA_UINT8);
  TEST_OCCA_TYPE((int16_t) 1, OCCA_INT16);
  TEST_OCCA_TYPE((uint16_t) 1, OCCA_UINT16);
  TEST_OCCA_TYPE((int32_t) 1, OCCA_INT32);
  TEST_OCCA_TYPE((uint32_t) 1, OCCA_UINT32);
  TEST_OCCA_TYPE((int64_t) 1, OCCA_INT64);
  TEST_OCCA_TYPE((uint64_t) 1, OCCA_UINT64);
  TEST_OCCA_TYPE((float) 1.0, OCCA_FLOAT);
  TEST_OCCA_TYPE((double) 1.0, OCCA_DOUBLE);

  TEST_OCCA_TYPE(occa::device(), OCCA_DEVICE);
  TEST_OCCA_TYPE(occa::kernel(), OCCA_KERNEL);
  TEST_OCCA_TYPE(occa::memory(), OCCA_MEMORY);
  TEST_OCCA_TYPE(*(new occa::properties()), OCCA_PROPERTIES);

  ASSERT_THROW_START {
    occa::c::newOccaType<std::string>("hi");
  } ASSERT_THROW_END;

#undef TEST_OCCA_TYPE
}
