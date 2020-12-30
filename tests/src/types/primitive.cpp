#include <occa/defines.hpp>
#include <occa/types/primitive.hpp>
#include <occa/internal/utils/sys.hpp>
#include <occa/internal/utils/testing.hpp>

void testInit();
void testLoad();
void testBadParsing();
void testToString();

int main(const int argc, const char **argv) {
  testInit();
  testLoad();
  testBadParsing();
  testToString();

  return 0;
}

void testInit() {
  ASSERT_EQ(true,
            (bool) occa::primitive(true));
  ASSERT_EQ(false,
            (bool) occa::primitive(false));

  ASSERT_EQ((int8_t) 15,
            (int8_t) occa::primitive((int8_t) 15));
  ASSERT_EQ((int16_t) 15,
            (int16_t) occa::primitive((int16_t) 15));
  ASSERT_EQ((int32_t) 15,
            (int32_t) occa::primitive((int32_t) 15));
  ASSERT_EQ((int64_t) 15,
            (int64_t) occa::primitive((int64_t) 15));

  ASSERT_EQ((uint8_t) 15,
            (uint8_t) occa::primitive((uint8_t) 15));
  ASSERT_EQ((uint16_t) 15,
            (uint16_t) occa::primitive((uint16_t) 15));
  ASSERT_EQ((uint32_t) 15,
            (uint32_t) occa::primitive((uint32_t) 15));
  ASSERT_EQ((uint64_t) 15,
            (uint64_t) occa::primitive((uint64_t) 15));

  ASSERT_EQ((float) 1e-16,
            (float) occa::primitive((float) 1e-16));
  ASSERT_EQ((double) 1e-16,
            (double) occa::primitive((double) 1e-16));
}

void testLoad() {
  ASSERT_EQ(15,
            (int) occa::primitive("15"));
  ASSERT_EQ(-15,
            (int) occa::primitive("-15"));

  ASSERT_EQ(15,
            (int) occa::primitive("0xF"));
  ASSERT_EQ(15,
            (int) occa::primitive("0XF"));
  ASSERT_EQ(-15,
            (int) occa::primitive("-0xF"));
  ASSERT_EQ(-15,
            (int) occa::primitive("-0XF"));

  ASSERT_EQ(15,
            (int) occa::primitive("0b1111"));
  ASSERT_EQ(15,
            (int) occa::primitive("0B1111"));
  ASSERT_EQ(-15,
            (int) occa::primitive("-0b1111"));
  ASSERT_EQ(-15,
            (int) occa::primitive("-0B1111"));

  ASSERT_EQ(15.01,
            (double) occa::primitive("15.01"));
  ASSERT_EQ(-15.01,
            (double) occa::primitive("-15.01"));

  ASSERT_EQ(1e-16,
            (double) occa::primitive("1e-16"));
  ASSERT_EQ(1.e-16,
            (double) occa::primitive("1.e-16"));
  ASSERT_EQ(15.01,
            (double) occa::primitive("1.501e1"));
  ASSERT_EQ(-15.01,
            (double) occa::primitive("-1.501e1"));
  ASSERT_EQ(15.01,
            (double) occa::primitive("1.501E1"));
  ASSERT_EQ(-15.01,
            (double) occa::primitive("-1.501E1"));

  ASSERT_EQ(1e-15,
            (double) occa::primitive("1e-15"));
  ASSERT_EQ(1.e-15,
            (double) occa::primitive("1.e-15"));
  ASSERT_EQ(15.01,
            (double) occa::primitive("1.501e+1"));
  ASSERT_EQ(-15.01,
            (double) occa::primitive("-1.501e+1"));
  ASSERT_EQ(15.01,
            (double) occa::primitive("1.501E+1"));
  ASSERT_EQ(-15.01,
            (double) occa::primitive("-1.501E+1"));

  ASSERT_EQ(15.01,
            (double) occa::primitive("150.1e-1"));
  ASSERT_EQ(-15.01,
            (double) occa::primitive("-150.1e-1"));
  ASSERT_EQ(15.01,
            (double) occa::primitive("150.1E-1"));
  ASSERT_EQ(-15.01,
            (double) occa::primitive("-150.1E-1"));

  ASSERT_TRUE(!!(occa::primitive("0x7f800000U").type & occa::primitiveType::isUnsigned));

  ASSERT_TRUE(!!(occa::primitive("0x7f800000L").type & occa::primitiveType::int64_));
  ASSERT_TRUE(!!(occa::primitive("0x7f800000LL").type & occa::primitiveType::int64_));

  ASSERT_TRUE(!!(occa::primitive("0x7f800000UL").type & occa::primitiveType::uint64_));
  ASSERT_TRUE(!!(occa::primitive("0x7f800000LLU").type & occa::primitiveType::uint64_));
}

void testBadParsing() {
  ASSERT_EQ(occa::primitiveType::none,
            occa::primitive("").type);

  ASSERT_EQ(occa::primitiveType::none,
            occa::primitive(" ").type);

  ASSERT_EQ(occa::primitiveType::none,
            occa::primitive("-").type);

  ASSERT_EQ(occa::primitiveType::none,
            occa::primitive("+").type);

  ASSERT_EQ(occa::primitiveType::none,
            occa::primitive("-   ").type);

  ASSERT_EQ(occa::primitiveType::none,
            occa::primitive("+   ").type);

  ASSERT_EQ(occa::primitiveType::none,
            occa::primitive("0x").type);

  ASSERT_EQ(occa::primitiveType::none,
            occa::primitive("0b").type);

  ASSERT_EQ(occa::primitiveType::none,
            occa::primitive("A").type);

  ASSERT_EQ(occa::primitiveType::none,
            occa::primitive("*").type);
}

void testToString() {
  ASSERT_EQ("0xFFFFFFFFF",
            occa::primitive("0xFFFFFFFFF").toString());

  ASSERT_EQ("0x7f800000U",
            occa::primitive("0x7f800000U").toString());

  ASSERT_EQ("0x7f800000L",
            occa::primitive("0x7f800000L").toString());

  ASSERT_EQ("0x7f800000LL",
            occa::primitive("0x7f800000LL").toString());

  ASSERT_EQ("NaN",
            occa::primitive("").toString());
}
