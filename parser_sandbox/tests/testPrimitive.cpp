#include "occa/defines.hpp"
#include "occa/tools/sys.hpp"
#include "occa/tools/testing.hpp"
#include "primitive.hpp"

void testLoad();
void testToString();

int main(const int argc, const char **argv) {
  testLoad();
  testToString();
}

void testLoad() {
  occa::testing::compare<int>(occa::primitive("15"), 15);
  occa::testing::compare<int>(occa::primitive("-15"), -15);

  occa::testing::compare<int>(occa::primitive("0xF"), 15);
  occa::testing::compare<int>(occa::primitive("0XF"), 15);
  occa::testing::compare<int>(occa::primitive("-0xF"), -15);
  occa::testing::compare<int>(occa::primitive("-0XF"), -15);

  occa::testing::compare<int>(occa::primitive("0b1111"), 15);
  occa::testing::compare<int>(occa::primitive("0B1111"), 15);
  occa::testing::compare<int>(occa::primitive("-0b1111"), -15);
  occa::testing::compare<int>(occa::primitive("-0B1111"), -15);

  occa::testing::compare<double>(occa::primitive("15.01"), 15.01);
  occa::testing::compare<double>(occa::primitive("-15.01"), -15.01);

  occa::testing::compare<double>(occa::primitive("1.501e1"), 15.01);
  occa::testing::compare<double>(occa::primitive("-1.501e1"), -15.01);
  occa::testing::compare<double>(occa::primitive("1.501E1"), 15.01);
  occa::testing::compare<double>(occa::primitive("-1.501E1"), -15.01);

  occa::testing::compare<double>(occa::primitive("1.501e+1"), 15.01);
  occa::testing::compare<double>(occa::primitive("-1.501e+1"), -15.01);
  occa::testing::compare<double>(occa::primitive("1.501E+1"), 15.01);
  occa::testing::compare<double>(occa::primitive("-1.501E+1"), -15.01);

  occa::testing::compare<double>(occa::primitive("150.1e-1"), 15.01);
  occa::testing::compare<double>(occa::primitive("-150.1e-1"), -15.01);
  occa::testing::compare<double>(occa::primitive("150.1E-1"), 15.01);
  occa::testing::compare<double>(occa::primitive("-150.1E-1"), -15.01);
}

void testToString() {
  occa::testing::compare<std::string>(occa::primitive("0xFFFFFFFFF"), "68719476735L");
}
