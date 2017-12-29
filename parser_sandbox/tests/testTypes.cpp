#include "occa/tools/env.hpp"
#include "occa/tools/testing.hpp"

#include "type.hpp"
#include "typeBuiltins.hpp"
#include "expression.hpp"

void testFunction();
void testCasting();

int main(const int argc, const char **argv) {
  testFunction();
  testCasting();
}

using namespace occa::lang;

void testFunction() {
  qualifiers_t q1;
  q1.add(volatile_);

  type_t t1_0(float_);
  t1_0.addQualifier(const_);
  pointerType t1_1(const_, t1_0);
  referenceType t1(t1_1);
  pointerType t2(t1_1);
  typedefType td1(t1, "t1");
  typedefType td2(t2, "t2");

  pointerType arg3(volatile_, char_);

  primitiveNode arg4Size(1337);
  arrayType arg4(t2, arg4Size);

  functionType f(void_, "foo");
  f.addArgument(t1 , "a");
  f.addArgument(td2, "b");
  f.addArgument(volatile_, float_, "c");
  f.addArgument(arg3);
  f.addArgument(arg4, "array");
  f.addArgument(double_, "e");

  functionType f2(f, "bar");

  std::cout << "q1   = " << q1.toString() << '\n'
            << "t1_0 = " << t1_0.toString() << '\n'
            << "t1_1 = " << t1_1.toString() << '\n'
            << "t1   = " << t1.toString() << '\n'
            << "t2   = " << t2.toString() << '\n'
            << "td1  = " << td1.declarationToString() << '\n'
            << "td2  = " << td2.declarationToString() << '\n'
            << "f    =\n" << f.declarationToString() << '\n'
            << "f2   =\n" << f2.declarationToString() << '\n';
}

void testCasting() {
  // All primitive can be cast to each other
  const primitiveType* types[10] = {
    &bool_,
    &char_, &char16_t_, &char32_t_, &wchar_t_,
    &short_,
    &int_, &long_,
    &float_, &double_ };
  for (int j = 0; j < 10; ++j) {
    const primitiveType &jType = *types[j];
    for (int i = 0; i < 10; ++i) {
      const primitiveType &iType = *types[i];
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
  primitiveNode one(1);
  type_t constInt(const_, int_);
  pointerType intPointer(int_);
  arrayType intArray(int_);
  arrayType intArray2(int_, one);
  pointerType constIntArray(const_, constInt);
  pointerType constIntArray2(constInt);

  std::cout << "intPointer    : " << intPointer.toString() << '\n'
            << "intArray      : " << intArray.toString() << '\n'
            << "intArray2     : " << intArray2.toString() << '\n'
            << "constIntArray : " << constIntArray.toString() << '\n'
            << "constIntArray2: " << constIntArray2.toString() << '\n';

  // Test explicit casting
  OCCA_TEST_COMPARE(intPointer.canBeCastedToExplicitly(intArray),
                    true);
  OCCA_TEST_COMPARE(intPointer.canBeCastedToExplicitly(intArray2),
                    true);
  OCCA_TEST_COMPARE(intPointer.canBeCastedToExplicitly(constIntArray),
                    true);
  OCCA_TEST_COMPARE(intPointer.canBeCastedToExplicitly(constIntArray),
                    true);

  OCCA_TEST_COMPARE(intArray.canBeCastedToExplicitly(intPointer),
                    true);
  OCCA_TEST_COMPARE(intArray.canBeCastedToExplicitly(intArray2),
                    true);
  OCCA_TEST_COMPARE(intArray.canBeCastedToExplicitly(constIntArray),
                    true);
  OCCA_TEST_COMPARE(intArray.canBeCastedToExplicitly(constIntArray2),
                    true);

  OCCA_TEST_COMPARE(intArray2.canBeCastedToExplicitly(intPointer),
                    true);
  OCCA_TEST_COMPARE(intArray2.canBeCastedToExplicitly(intArray),
                    true);
  OCCA_TEST_COMPARE(intArray2.canBeCastedToExplicitly(constIntArray),
                    true);
  OCCA_TEST_COMPARE(intArray2.canBeCastedToExplicitly(constIntArray2),
                    true);

  OCCA_TEST_COMPARE(constIntArray.canBeCastedToExplicitly(intPointer),
                    true);
  OCCA_TEST_COMPARE(constIntArray.canBeCastedToExplicitly(intArray),
                    true);
  OCCA_TEST_COMPARE(constIntArray.canBeCastedToExplicitly(intArray2),
                    true);
  OCCA_TEST_COMPARE(constIntArray.canBeCastedToExplicitly(constIntArray2),
                    true);

  OCCA_TEST_COMPARE(constIntArray2.canBeCastedToExplicitly(intPointer),
                    true);
  OCCA_TEST_COMPARE(constIntArray2.canBeCastedToExplicitly(intArray),
                    true);
  OCCA_TEST_COMPARE(constIntArray2.canBeCastedToExplicitly(intArray2),
                    true);
  OCCA_TEST_COMPARE(constIntArray2.canBeCastedToExplicitly(constIntArray),
                    true);

  // Test implicit casting
  OCCA_TEST_COMPARE(intPointer.canBeCastedToImplicitly(intArray),
                    true);
  OCCA_TEST_COMPARE(intPointer.canBeCastedToImplicitly(intArray2),
                    true);
  OCCA_TEST_COMPARE(intPointer.canBeCastedToImplicitly(constIntArray),
                    false);
  OCCA_TEST_COMPARE(intPointer.canBeCastedToImplicitly(constIntArray2),
                    false);

  OCCA_TEST_COMPARE(intArray.canBeCastedToImplicitly(intPointer),
                    true);
  OCCA_TEST_COMPARE(intArray.canBeCastedToImplicitly(intArray2),
                    true);
  OCCA_TEST_COMPARE(intArray.canBeCastedToImplicitly(constIntArray),
                    false);
  OCCA_TEST_COMPARE(intArray.canBeCastedToImplicitly(constIntArray2),
                    false);

  OCCA_TEST_COMPARE(intArray2.canBeCastedToImplicitly(intPointer),
                    true);
  OCCA_TEST_COMPARE(intArray2.canBeCastedToImplicitly(intArray),
                    true);
  OCCA_TEST_COMPARE(intArray2.canBeCastedToImplicitly(constIntArray),
                    false);
  OCCA_TEST_COMPARE(intArray2.canBeCastedToImplicitly(constIntArray2),
                    false);

  OCCA_TEST_COMPARE(constIntArray.canBeCastedToImplicitly(intPointer),
                    false);
  OCCA_TEST_COMPARE(constIntArray.canBeCastedToImplicitly(intArray),
                    false);
  OCCA_TEST_COMPARE(constIntArray.canBeCastedToImplicitly(intArray2),
                    false);
  OCCA_TEST_COMPARE(constIntArray.canBeCastedToImplicitly(constIntArray2),
                    true);

  OCCA_TEST_COMPARE(constIntArray2.canBeCastedToImplicitly(intPointer),
                    false);
  OCCA_TEST_COMPARE(constIntArray2.canBeCastedToImplicitly(intArray),
                    false);
  OCCA_TEST_COMPARE(constIntArray2.canBeCastedToImplicitly(intArray2),
                    false);
  OCCA_TEST_COMPARE(constIntArray2.canBeCastedToImplicitly(constIntArray),
                    true);
}
