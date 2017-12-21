#include "occa/tools/env.hpp"
#include "occa/tools/testing.hpp"

#include "type.hpp"
#include "typeBuiltins.hpp"
#include "expression.hpp"

void testFunction();

int main(const int argc, const char **argv) {
  testFunction();
}

using namespace occa::lang;

void testFunction() {
  qualifiers q1;
  q1.add(volatile_);

  type_t t1_0(float_);
  t1_0.addQualifier(const_);
  pointerType t1_1(t1_0, const_);
  referenceType t1(t1_1);
  pointerType t2(t1_1);
  typedefType td1(t1, "t1");
  typedefType td2(t2, "t2");

  pointerType arg3(char_, volatile_);

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
