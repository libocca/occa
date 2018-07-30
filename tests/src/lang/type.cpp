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
#include <occa/tools/env.hpp>
#include <occa/tools/testing.hpp>

#include <occa/lang/token.hpp>
#include <occa/lang/type.hpp>
#include <occa/lang/expression.hpp>
#include <occa/lang/builtins/types.hpp>
#include <occa/lang/variable.hpp>

void testQualifiers();
void testCasting();
void testComparision();

int main(const int argc, const char **argv) {
  testQualifiers();
  testCasting();
  testComparision();

  return 0;
}

using namespace occa::lang;

void testQualifiers() {
  qualifiers_t q1, q2;
  for (int i = 0; i < 10; ++i) {
    q1.add(const_);
    q1.addFirst(extern_);
    q2.addFirst(externC);
  }
  ASSERT_EQ(2, q1.size());
  ASSERT_EQ(1, q2.size());
}

// TODO: Reimplement casting checking
void testCasting() {
#if 0
  // All primitive can be cast to each other
  const primitive_t* types[9] = {
    &bool_,
    &char_, &char16_t_, &char32_t_, &wchar_t_,
    &short_,
    &int_,
    &float_, &double_
  };
  for (int j = 0; j < 9; ++j) {
    const primitive_t &jType = *types[j];
    for (int i = 0; i < 9; ++i) {
      const primitive_t &iType = *types[i];
      OCCA_ERROR("Oops, could not cast explicitly ["
                 << jType.uniqueName() << "] to ["
                 << iType.uniqueName() << "]",
                 jType.canBeCastedToExplicitly(iType));
      OCCA_ERROR("Oops, could not cast implicitly ["
                 << jType.uniqueName() << "] to ["
                 << iType.uniqueName() << "]",
                 jType.canBeCastedToImplicitly(iType));
    }
  }

  // Test pointer <-> array
  primitiveNode one(NULL, 1);
  type_t constInt(const_, int_);
  pointer_t intPointer(int_);
  array_t intArray(int_);
  array_t intArray2(int_, one);
  pointer_t constIntArray(const_, constInt);
  pointer_t constIntArray2(constInt);

  std::cout << "intPointer    : " << intPointer.toString() << '\n'
            << "intArray      : " << intArray.toString() << '\n'
            << "intArray2     : " << intArray2.toString() << '\n'
            << "constIntArray : " << constIntArray.toString() << '\n'
            << "constIntArray2: " << constIntArray2.toString() << '\n';

  // Test explicit casting
  ASSERT_TRUE(intPointer.canBeCastedToExplicitly(intArray));
  ASSERT_TRUE(intPointer.canBeCastedToExplicitly(intArray2));
  ASSERT_TRUE(intPointer.canBeCastedToExplicitly(constIntArray));
  ASSERT_TRUE(intPointer.canBeCastedToExplicitly(constIntArray));

  ASSERT_TRUE(intArray.canBeCastedToExplicitly(intPointer));
  ASSERT_TRUE(intArray.canBeCastedToExplicitly(intArray2));
  ASSERT_TRUE(intArray.canBeCastedToExplicitly(constIntArray));
  ASSERT_TRUE(intArray.canBeCastedToExplicitly(constIntArray2));

  ASSERT_TRUE(intArray2.canBeCastedToExplicitly(intPointer));
  ASSERT_TRUE(intArray2.canBeCastedToExplicitly(intArray));
  ASSERT_TRUE(intArray2.canBeCastedToExplicitly(constIntArray));
  ASSERT_TRUE(intArray2.canBeCastedToExplicitly(constIntArray2));

  ASSERT_TRUE(constIntArray.canBeCastedToExplicitly(intPointer));
  ASSERT_TRUE(constIntArray.canBeCastedToExplicitly(intArray));
  ASSERT_TRUE(constIntArray.canBeCastedToExplicitly(intArray2));
  ASSERT_TRUE(constIntArray.canBeCastedToExplicitly(constIntArray2));

  ASSERT_TRUE(constIntArray2.canBeCastedToExplicitly(intPointer));
  ASSERT_TRUE(constIntArray2.canBeCastedToExplicitly(intArray));
  ASSERT_TRUE(constIntArray2.canBeCastedToExplicitly(intArray2));
  ASSERT_TRUE(constIntArray2.canBeCastedToExplicitly(constIntArray));

  // Test implicit casting
  ASSERT_TRUE(intPointer.canBeCastedToImplicitly(intArray));
  ASSERT_TRUE(intPointer.canBeCastedToImplicitly(intArray2));
  ASSERT_FALSE(intPointer.canBeCastedToImplicitly(constIntArray));
  ASSERT_FALSE(intPointer.canBeCastedToImplicitly(constIntArray2));

  ASSERT_TRUE(intArray.canBeCastedToImplicitly(intPointer));
  ASSERT_TRUE(intArray.canBeCastedToImplicitly(intArray2));
  ASSERT_FALSE(intArray.canBeCastedToImplicitly(constIntArray));
  ASSERT_FALSE(intArray.canBeCastedToImplicitly(constIntArray2));

  ASSERT_TRUE(intArray2.canBeCastedToImplicitly(intPointer));
  ASSERT_TRUE(intArray2.canBeCastedToImplicitly(intArray));
  ASSERT_FALSE(intArray2.canBeCastedToImplicitly(constIntArray));
  ASSERT_FALSE(intArray2.canBeCastedToImplicitly(constIntArray2));

  ASSERT_FALSE(constIntArray.canBeCastedToImplicitly(intPointer));
  ASSERT_FALSE(constIntArray.canBeCastedToImplicitly(intArray));
  ASSERT_FALSE(constIntArray.canBeCastedToImplicitly(intArray2));
  ASSERT_TRUE(constIntArray.canBeCastedToImplicitly(constIntArray2));

  ASSERT_FALSE(constIntArray2.canBeCastedToImplicitly(intPointer));
  ASSERT_FALSE(constIntArray2.canBeCastedToImplicitly(intArray));
  ASSERT_FALSE(constIntArray2.canBeCastedToImplicitly(intArray2));
  ASSERT_TRUE(constIntArray2.canBeCastedToImplicitly(constIntArray));
#endif
}

void testComparision() {
  // Test primitives
  const primitive_t* types[9] = {
    &bool_,
    &char_, &char16_t_, &char32_t_, &wchar_t_,
    &short_,
    &int_,
    &float_, &double_
  };
  for (int j = 0; j < 9; ++j) {
    vartype_t jVar(*types[j]);
    for (int i = 0; i < 9; ++i) {
      vartype_t iVar(*types[i]);
      ASSERT_EQ(i == j,
                iVar == jVar);
    }
  }

  // Test wrapped types
  identifierToken fooName(fileOrigin(),
                          "foo");
  vartype_t fakeFloat(float_);
  typedef_t typedefFloat(float_, fooName);
  ASSERT_TRUE(fakeFloat == typedefFloat);

  // Test qualifiers
  qualifiers_t q1, q2;
  q1 += const_;
  q1 += volatile_;
  q2 += volatile_;

  vartype_t qType1(float_);
  qType1 += const_;
  qType1 += volatile_;
  vartype_t qType2(float_);
  qType2 += volatile_;
  ASSERT_TRUE(qType1 != qType2);

  qType2 += const_;
  ASSERT_TRUE(qType1 == qType2);
}
