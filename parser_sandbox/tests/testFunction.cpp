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

  type &t1_0 = float_.clone();
  t1_0.add(const_);
  pointer t1_1(q1, t1_0);
  pointer t1(const_, t1_1);
  pointer t2(t1_1);
  // typedefType t2 = typedefType(const_, double_, "t2");

  // function f(void_, "foo");
  // f.addArgument(t1, "a");
  // f.addArgument(t2, "b");

  std::cout << "q1   = " << q1 << '\n'
            << "t1_0 = " << t1_0 << '\n'
            << "t1_1 = " << t1_1 << '\n'
            << "t1   = " << t1 << '\n'
            << "t2   = " << t2 << '\n';

  // std::cout << "q1   = " << q1 << '\n'
  //           << "t1_1 = " << t1_1 << '\n'
  //           << "t1   = " << t1 << '\n'
  //           << "t2   = " << t2 << '\n'
  //           << "f    = " << f << '\n';
}
