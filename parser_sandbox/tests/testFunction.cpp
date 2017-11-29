#include "occa/tools/env.hpp"
#include "occa/tools/testing.hpp"

#include "type.hpp"
#include "typeBuiltins.hpp"

void testFunction();

int main(const int argc, const char **argv) {
  testFunction();
}

using namespace occa::lang;

void testFunction() {
  qualifiers q1;
  q1.add(volatile_);

  type_t t1_0(float_);
  t1_0.add(const_);
  pointerType t1_1(q1, t1_0);
  referenceType t1(t1_1);
  pointerType t2(const_, t1_1);
  typedefType td1 = typedefType(t1, "t1");
  typedefType td2 = typedefType(t2, "t2");

  functionType f(void_, "foo");
  f.addArg(t1     , "a");
  f.addArg(td2    , "b");
  f.addArg(volatile_, float_ , "c");
  f.addArg(pointerType(const_, char_));
  f.addArg(double_, "e");

  std::cout << "q1   = " << q1.toString() << '\n'
            << "t1_0 = " << t1_0.toString() << '\n'
            << "t1_1 = " << t1_1.toString() << '\n'
            << "t1   = " << t1.toString() << '\n'
            << "t2   = " << t2.toString() << '\n'
            << "td1  = " << td1.toString() << '\n'
            << "td2  = " << td2.toString() << '\n'
            << "f    =\n" << f.toString() << '\n';
}
