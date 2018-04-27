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
#include "occa/tools/env.hpp"
#include "occa/tools/testing.hpp"

#include "token.hpp"
#include "type.hpp"
#include "expression.hpp"
#include "builtins/types.hpp"

void testBitfields();
void testFunction();
void testCasting();
void testComparision();

int main(const int argc, const char **argv) {
  testBitfields();
  testFunction();
  testCasting();
  testComparision();

  return 0;
}

using namespace occa::lang;

void testBitfields() {
  occa::bitfield bf1(0, 1 << 0);
  OCCA_ASSERT_EQUAL_BINARY(bf1.b1, 0UL);
  OCCA_ASSERT_EQUAL_BINARY(bf1.b2, 1UL);

  bf1 <<= (occa::bitfield::bits() / 2);
  OCCA_ASSERT_EQUAL_BINARY(bf1.b1, 1UL);
  OCCA_ASSERT_EQUAL_BINARY(bf1.b2, 0UL);

  bf1 >>= (occa::bitfield::bits() / 2);
  OCCA_ASSERT_EQUAL_BINARY(bf1.b1, 0UL);
  OCCA_ASSERT_EQUAL_BINARY(bf1.b2, 1UL);

  occa::bitfield bf2 = (occa::bitfield(0, 1 << 0) |
                        occa::bitfield(0, 1 << 1));

  OCCA_ASSERT_TRUE(bf1 & bf2);
  bf2 <<= 1;
  OCCA_ASSERT_FALSE(bf1 & bf2);

  const occa::bitfield a1(0, 1L << 0);
  const occa::bitfield a2(0, 1L << 1);
  const occa::bitfield b1(0, 1L << 2);
  const occa::bitfield b2(0, 1L << 3);
  const occa::bitfield c1(0, 1L << 4);
  const occa::bitfield c2(0, 1L << 5);

  const occa::bitfield a = (a1 | a2);
  const occa::bitfield b = (b1 | b2);
  const occa::bitfield c = (c1 | c2);

  const occa::bitfield start = (a1 | b1 | c1);
  const occa::bitfield end   = (a2 | b2 | c2);

  OCCA_ASSERT_TRUE(a & a1);
  OCCA_ASSERT_TRUE(a & a2);

  OCCA_ASSERT_TRUE(start & a);
  OCCA_ASSERT_TRUE(start & a1);
  OCCA_ASSERT_TRUE(start & b1);
  OCCA_ASSERT_TRUE(start & c1);

  OCCA_ASSERT_TRUE(end & a);
  OCCA_ASSERT_TRUE(end & a2);
  OCCA_ASSERT_TRUE(end & b2);
  OCCA_ASSERT_TRUE(end & c2);

  OCCA_ASSERT_FALSE(a & b);
  OCCA_ASSERT_FALSE(a & c);
  OCCA_ASSERT_FALSE(b & c);

  OCCA_ASSERT_FALSE(start & end);

  OCCA_ASSERT_TRUE(a1 != a2);
  OCCA_ASSERT_TRUE(a1 <  a2);
  OCCA_ASSERT_TRUE(a2 <= a2);
  OCCA_ASSERT_TRUE(a2 == a2);
  OCCA_ASSERT_TRUE(a2 >= a2);
  OCCA_ASSERT_TRUE(a2 >  a1);
}

void testFunction() {
  qualifiers_t q1;
  q1 += (volatile_);

  vartype_t t1_0(float_);
  t1_0 += const_;

  vartype_t t1_1 = t1_0;
  t1_1 += const_;
  t1_1 += pointer_t();

  vartype_t t1 = t1_1;
  unknownToken dummyReferenceToken((fileOrigin()));
  t1.setReferenceToken(&dummyReferenceToken);

  vartype_t t2 = t1_1;
  qualifiers_t t2q;
  t2q += const_;
  t2 += pointer_t(t2q);

  identifierToken td1Name(fileOrigin(),
                          "td1");
  identifierToken td2Name(fileOrigin(),
                          "td2");
  typedef_t td1(t1, td1Name);
  typedef_t td2(t2, td2Name);

  vartype_t arg3(char_);
  arg3 += volatile_;

  primitiveNode arg4Size(NULL, 1337);
  vartype_t arg4(t2);

  operatorToken arg4Start(fileOrigin(),
                          op::bracketStart);
  operatorToken arg4End(fileOrigin(),
                        op::bracketStart);
  arg4 += array_t(arg4Start,
                  arg4End,
                  &arg4Size.clone());

  function_t f(void_, "foo");
  identifierToken a(fileOrigin(), "a");
  identifierToken b(fileOrigin(), "b");
  identifierToken array(fileOrigin(), "array");
  identifierToken e(fileOrigin(), "e");
  f += variable(t1 , &a);
  f += variable(td2, &b);
  f += variable(arg3);
  f += variable(arg4, &array);
  f += variable(double_, &e);

  function_t f2(f, "bar");

  printer pout(std::cerr);

  pout << "q1   = " << q1 << '\n'
       << "t1_0 = " << t1_0 << '\n'
       << "t1_1 = " << t1_1 << '\n'
       << "t1   = " << t1 << '\n'
       << "t2   = " << t2 << '\n';

  pout << "td1  = ";
  td1.printDeclaration(pout);
  pout << '\n';

  pout << "td2  = ";
  td2.printDeclaration(pout);
  pout << '\n';

  f.printDeclaration(pout);
  pout << '\n';

  f2.printDeclaration(pout);
  pout << '\n';

  f.isPointer = f2.isPointer = true;
  f.printDeclaration(pout);
  pout << '\n';

  f2.printDeclaration(pout);
  pout << '\n';

  f.isPointer = f2.isPointer = false;
  f.isBlock   = f2.isBlock   = true;
  f.printDeclaration(pout);
  pout << '\n';

  f2.printDeclaration(pout);
  pout << '\n';
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
  OCCA_ASSERT_TRUE(intPointer.canBeCastedToExplicitly(intArray));
  OCCA_ASSERT_TRUE(intPointer.canBeCastedToExplicitly(intArray2));
  OCCA_ASSERT_TRUE(intPointer.canBeCastedToExplicitly(constIntArray));
  OCCA_ASSERT_TRUE(intPointer.canBeCastedToExplicitly(constIntArray));

  OCCA_ASSERT_TRUE(intArray.canBeCastedToExplicitly(intPointer));
  OCCA_ASSERT_TRUE(intArray.canBeCastedToExplicitly(intArray2));
  OCCA_ASSERT_TRUE(intArray.canBeCastedToExplicitly(constIntArray));
  OCCA_ASSERT_TRUE(intArray.canBeCastedToExplicitly(constIntArray2));

  OCCA_ASSERT_TRUE(intArray2.canBeCastedToExplicitly(intPointer));
  OCCA_ASSERT_TRUE(intArray2.canBeCastedToExplicitly(intArray));
  OCCA_ASSERT_TRUE(intArray2.canBeCastedToExplicitly(constIntArray));
  OCCA_ASSERT_TRUE(intArray2.canBeCastedToExplicitly(constIntArray2));

  OCCA_ASSERT_TRUE(constIntArray.canBeCastedToExplicitly(intPointer));
  OCCA_ASSERT_TRUE(constIntArray.canBeCastedToExplicitly(intArray));
  OCCA_ASSERT_TRUE(constIntArray.canBeCastedToExplicitly(intArray2));
  OCCA_ASSERT_TRUE(constIntArray.canBeCastedToExplicitly(constIntArray2));

  OCCA_ASSERT_TRUE(constIntArray2.canBeCastedToExplicitly(intPointer));
  OCCA_ASSERT_TRUE(constIntArray2.canBeCastedToExplicitly(intArray));
  OCCA_ASSERT_TRUE(constIntArray2.canBeCastedToExplicitly(intArray2));
  OCCA_ASSERT_TRUE(constIntArray2.canBeCastedToExplicitly(constIntArray));

  // Test implicit casting
  OCCA_ASSERT_TRUE(intPointer.canBeCastedToImplicitly(intArray));
  OCCA_ASSERT_TRUE(intPointer.canBeCastedToImplicitly(intArray2));
  OCCA_ASSERT_FALSE(intPointer.canBeCastedToImplicitly(constIntArray));
  OCCA_ASSERT_FALSE(intPointer.canBeCastedToImplicitly(constIntArray2));

  OCCA_ASSERT_TRUE(intArray.canBeCastedToImplicitly(intPointer));
  OCCA_ASSERT_TRUE(intArray.canBeCastedToImplicitly(intArray2));
  OCCA_ASSERT_FALSE(intArray.canBeCastedToImplicitly(constIntArray));
  OCCA_ASSERT_FALSE(intArray.canBeCastedToImplicitly(constIntArray2));

  OCCA_ASSERT_TRUE(intArray2.canBeCastedToImplicitly(intPointer));
  OCCA_ASSERT_TRUE(intArray2.canBeCastedToImplicitly(intArray));
  OCCA_ASSERT_FALSE(intArray2.canBeCastedToImplicitly(constIntArray));
  OCCA_ASSERT_FALSE(intArray2.canBeCastedToImplicitly(constIntArray2));

  OCCA_ASSERT_FALSE(constIntArray.canBeCastedToImplicitly(intPointer));
  OCCA_ASSERT_FALSE(constIntArray.canBeCastedToImplicitly(intArray));
  OCCA_ASSERT_FALSE(constIntArray.canBeCastedToImplicitly(intArray2));
  OCCA_ASSERT_TRUE(constIntArray.canBeCastedToImplicitly(constIntArray2));

  OCCA_ASSERT_FALSE(constIntArray2.canBeCastedToImplicitly(intPointer));
  OCCA_ASSERT_FALSE(constIntArray2.canBeCastedToImplicitly(intArray));
  OCCA_ASSERT_FALSE(constIntArray2.canBeCastedToImplicitly(intArray2));
  OCCA_ASSERT_TRUE(constIntArray2.canBeCastedToImplicitly(constIntArray));
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
      OCCA_ASSERT_EQUAL(i == j,
                        iVar == jVar);
    }
  }

  // Test wrapped types
  identifierToken fooName(fileOrigin(),
                          "foo");
  vartype_t fakeFloat(float_);
  typedef_t typedefFloat(float_, fooName);
  OCCA_ASSERT_TRUE(fakeFloat == typedefFloat);

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
  OCCA_ASSERT_TRUE(qType1 != qType2);

  qType2 += const_;
  OCCA_ASSERT_TRUE(qType1 == qType2);
}
