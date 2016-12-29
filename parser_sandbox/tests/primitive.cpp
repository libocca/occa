#include "/Users/dsm5/git/night/examples/foo/tests/test.cpp"
#include "/Users/dsm5/git/night/examples/foo/primitive.hpp"

void testLoad();
void testToString();

int main(const int argc, const char **argv) {
  testLoad();
  testToString();
}

void testLoad() {
  test::compare<int>(primitive("15"), 15);
  test::compare<int>(primitive("-15"), -15);

  test::compare<int>(primitive("0xF"), 15);
  test::compare<int>(primitive("0XF"), 15);
  test::compare<int>(primitive("-0xF"), -15);
  test::compare<int>(primitive("-0XF"), -15);

  test::compare<int>(primitive("0b1111"), 15);
  test::compare<int>(primitive("0B1111"), 15);
  test::compare<int>(primitive("-0b1111"), -15);
  test::compare<int>(primitive("-0B1111"), -15);

  test::compare<double>(primitive("15.01"), 15.01);
  test::compare<double>(primitive("-15.01"), -15.01);

  test::compare<double>(primitive("1.501e1"), 15.01);
  test::compare<double>(primitive("-1.501e1"), -15.01);
  test::compare<double>(primitive("1.501E1"), 15.01);
  test::compare<double>(primitive("-1.501E1"), -15.01);

  test::compare<double>(primitive("1.501e+1"), 15.01);
  test::compare<double>(primitive("-1.501e+1"), -15.01);
  test::compare<double>(primitive("1.501E+1"), 15.01);
  test::compare<double>(primitive("-1.501E+1"), -15.01);

  test::compare<double>(primitive("150.1e-1"), 15.01);
  test::compare<double>(primitive("-150.1e-1"), -15.01);
  test::compare<double>(primitive("150.1E-1"), 15.01);
  test::compare<double>(primitive("-150.1E-1"), -15.01);
}

void testToString() {
  test::compare<std::string>(primitive("0xFFFFFFFFF"), "68719476735L");
}