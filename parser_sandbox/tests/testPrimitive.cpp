#include "occa/defines.hpp"
#include "occa/tools/sys.hpp"
#include "occa/tools/testing.hpp"
#include "occa/parser/primitive.hpp"

void testLoad();
void testToString();

int main(const int argc, const char **argv) {
  testLoad();
  testToString();
}

void testLoad() {
  OCCA_TEST_COMPARE(15,
                    (int) occa::primitive("15"));
  OCCA_TEST_COMPARE(-15,
                    (int) occa::primitive("-15"));

  OCCA_TEST_COMPARE(15,
                    (int) occa::primitive("0xF"));
  OCCA_TEST_COMPARE(15,
                    (int) occa::primitive("0XF"));
  OCCA_TEST_COMPARE(-15,
                    (int) occa::primitive("-0xF"));
  OCCA_TEST_COMPARE(-15,
                    (int) occa::primitive("-0XF"));

  OCCA_TEST_COMPARE(15,
                    (int) occa::primitive("0b1111"));
  OCCA_TEST_COMPARE(15,
                    (int) occa::primitive("0B1111"));
  OCCA_TEST_COMPARE(-15,
                    (int) occa::primitive("-0b1111"));
  OCCA_TEST_COMPARE(-15,
                    (int) occa::primitive("-0B1111"));

  OCCA_TEST_COMPARE(15.01,
                    (double) occa::primitive("15.01"));
  OCCA_TEST_COMPARE(-15.01,
                    (double) occa::primitive("-15.01"));

  OCCA_TEST_COMPARE(1e-16,
                    (double) occa::primitive("1e-16"));
  OCCA_TEST_COMPARE(15.01,
                    (double) occa::primitive("1.501e1"));
  OCCA_TEST_COMPARE(-15.01,
                    (double) occa::primitive("-1.501e1"));
  OCCA_TEST_COMPARE(15.01,
                    (double) occa::primitive("1.501E1"));
  OCCA_TEST_COMPARE(-15.01,
                    (double) occa::primitive("-1.501E1"));

  OCCA_TEST_COMPARE(15.01,
                    (double) occa::primitive("1.501e+1"));
  OCCA_TEST_COMPARE(-15.01,
                    (double) occa::primitive("-1.501e+1"));
  OCCA_TEST_COMPARE(15.01,
                    (double) occa::primitive("1.501E+1"));
  OCCA_TEST_COMPARE(-15.01,
                    (double) occa::primitive("-1.501E+1"));

  OCCA_TEST_COMPARE(15.01,
                    (double) occa::primitive("150.1e-1"));
  OCCA_TEST_COMPARE(-15.01,
                    (double) occa::primitive("-150.1e-1"));
  OCCA_TEST_COMPARE(15.01,
                    (double) occa::primitive("150.1E-1"));
  OCCA_TEST_COMPARE(-15.01,
                    (double) occa::primitive("-150.1E-1"));
}

void testToString() {
  OCCA_TEST_COMPARE("68719476735L",
                    (std::string) occa::primitive("0xFFFFFFFFF"));
}
