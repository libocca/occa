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

  type t1_0(float_);
  t1_0.add(const_);
  pointer t1_1(q1, t1_0);
  reference t1(t1_1);
  pointer t2(const_, t1_1);
  typedefType td1 = typedefType(t1, "t1");
  typedefType td2 = typedefType(t2, "t2");

  function f(void_, "foo");
  f.add(t1     , "a");
  f.add(td2    , "b");
  f.add(volatile_, float_ , "c");
  f.add(pointer(const_, char_));
  f.add(double_, "e");

  std::cout << "q1   = " << q1 << '\n'
            << "t1_0 = " << t1_0 << '\n'
            << "t1_1 = " << t1_1 << '\n'
            << "t1   = " << t1 << '\n'
            << "t2   = " << t2 << '\n'
            << "td1  = " << td1 << '\n'
            << "td2  = " << td2 << '\n'
            << "f    =\n" << f << '\n';
}
