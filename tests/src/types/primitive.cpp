#include <occa/defines.hpp>
#include <occa/types/primitive.hpp>
#include <occa/internal/utils/sys.hpp>
#include <occa/internal/utils/testing.hpp>
#include <limits>

void testInit();
void testLoad();
void testBadParsing();
void testToString();
void testSizeOf();
void testNot();
void testPositive();
void testNegative();
void testTilde();

int main(const int argc, const char **argv) {
  testInit();
  testLoad();
  testBadParsing();
  testToString();
  testSizeOf();
  testNot();
  testPositive();
  testNegative();
  testTilde();

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

  std::string fifteen{"15"};
  ASSERT_EQ(15, (int) occa::primitive(fifteen));
  ASSERT_EQ((int) -15, (int) occa::primitive::load("-15",true));

  ASSERT_EQ(15,
            (int) occa::primitive("0xF"));
  ASSERT_EQ(15,
            (int) occa::primitive("0XF"));
  ASSERT_EQ(-15,
            (int) occa::primitive("-0xF"));
  ASSERT_EQ(-15,
            (int) occa::primitive("-0XF"));

  ASSERT_EQ((uint8_t) 15,(uint8_t) occa::primitive("0b1111"));
  ASSERT_EQ((uint8_t) 15,(uint8_t) occa::primitive("0B1111"));
  ASSERT_EQ((int8_t) -15,(int8_t) occa::primitive("-0b1111"));
  ASSERT_EQ((int8_t) -15,(int8_t) occa::primitive("-0B1111"));

  ASSERT_EQ((uint16_t) 15,
    (uint16_t) occa::primitive("0b00001111"));
  ASSERT_EQ((uint16_t) 15,
    (uint16_t) occa::primitive("0B00001111"));
  ASSERT_EQ((int16_t) -15,
    (int16_t) occa::primitive("-0b00001111"));
  ASSERT_EQ((int16_t) -15,
    (int16_t) occa::primitive("-0B00001111"));

  ASSERT_EQ((uint32_t) 15,
    (uint32_t) occa::primitive("0b0000000000001111"));
  ASSERT_EQ((uint32_t) 15,
    (uint32_t) occa::primitive("0B0000000000001111"));
  ASSERT_EQ((int32_t) -15,
    (int32_t) occa::primitive("-0b0000000000001111"));
  ASSERT_EQ((int32_t) -15,
    (int32_t) occa::primitive("-0B0000000000001111"));

  ASSERT_EQ((uint64_t) 15,
    (uint64_t) occa::primitive("0b00000000000000000000000000001111"));
  ASSERT_EQ((uint64_t) 15,
    (uint64_t) occa::primitive("0B00000000000000000000000000001111"));
  ASSERT_EQ((int64_t) -15,
    (int64_t) occa::primitive("-0b00000000000000000000000000001111"));
  ASSERT_EQ((int64_t) -15,
    (int64_t) occa::primitive("-0B00000000000000000000000000001111"));

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
  
  ASSERT_EQ(15.01,(double) occa::primitive("15.01"));
  ASSERT_EQ(-15.01,(double) occa::primitive("-15.01"));

  ASSERT_EQ( (float) 1e-16  ,(float) occa::primitive("1e-16F"));
  ASSERT_EQ( (float) 1.e-16 ,(float) occa::primitive("1.e-16F"));
  ASSERT_EQ( (float) 15.01  ,(float) occa::primitive("1.501e1F"));
  ASSERT_EQ( (float) -15.01 ,(float) occa::primitive("-1.501e1F"));
  ASSERT_EQ( (float) 15.01  ,(float) occa::primitive("1.501E1F"));
  ASSERT_EQ( (float) -15.01 ,(float) occa::primitive("-1.501E1F"));
  ASSERT_EQ( (float) 1e-15  ,(float) occa::primitive("1e-15F"));
  ASSERT_EQ( (float) 1.e-15 ,(float) occa::primitive("1.e-15F"));
  ASSERT_EQ( (float) 15.01  ,(float) occa::primitive("1.501e+1F"));
  ASSERT_EQ( (float) -15.01 ,(float) occa::primitive("-1.501e+1F"));
  ASSERT_EQ( (float) 15.01  ,(float) occa::primitive("1.501E+1F"));
  ASSERT_EQ( (float) -15.01 ,(float) occa::primitive("-1.501E+1F"));
  ASSERT_EQ( (float) 15.01  ,(float) occa::primitive("150.1e-1F"));
  ASSERT_EQ( (float) -15.01 ,(float) occa::primitive("-150.1e-1F"));
  ASSERT_EQ( (float) 15.01  ,(float) occa::primitive("150.1E-1F"));
  ASSERT_EQ( (float) -15.01 ,(float) occa::primitive("-150.1E-1F"));
  
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

  ASSERT_EQ("1.2345f",
            occa::primitive("1.2345f").toString());

  ASSERT_EQ("",occa::primitive().toString());
}

void testSizeOf() {
  ASSERT_EQ(sizeof(bool),     occa::primitive(true)         .sizeof_()); 
  ASSERT_EQ(sizeof(uint8_t),  occa::primitive((uint8_t) 1)  .sizeof_());
  ASSERT_EQ(sizeof(uint16_t), occa::primitive((uint16_t) 1) .sizeof_());
  ASSERT_EQ(sizeof(uint32_t), occa::primitive((uint32_t) 1) .sizeof_());
  ASSERT_EQ(sizeof(uint64_t), occa::primitive((uint64_t) 1) .sizeof_());
  ASSERT_EQ(sizeof(int8_t),   occa::primitive((int8_t) 1)   .sizeof_());
  ASSERT_EQ(sizeof(int16_t),  occa::primitive((int16_t) 1)  .sizeof_());
  ASSERT_EQ(sizeof(int32_t),  occa::primitive((int32_t) 1)  .sizeof_());
  ASSERT_EQ(sizeof(int64_t),  occa::primitive((int64_t) 1)  .sizeof_());
  ASSERT_EQ(sizeof(float),    occa::primitive((float) 1)    .sizeof_());
  ASSERT_EQ(sizeof(double),   occa::primitive((double) 1)   .sizeof_());
}

void testNot() {
  ASSERT_EQ(false        , (bool)     occa::primitive::not_(true)         );
  ASSERT_EQ((uint8_t)   0, (uint8_t)  occa::primitive::not_((uint8_t)   1));
  ASSERT_EQ((uint16_t)  0, (uint16_t) occa::primitive::not_((uint16_t)  1));
  ASSERT_EQ((uint32_t)  0, (uint32_t) occa::primitive::not_((uint32_t)  1));
  ASSERT_EQ((uint64_t)  0, (uint64_t) occa::primitive::not_((uint64_t)  1));
  ASSERT_EQ((int8_t)    0, (int8_t)   occa::primitive::not_((int8_t)    1));
  ASSERT_EQ((int16_t)   0, (int16_t)  occa::primitive::not_((int16_t)   1));
  ASSERT_EQ((int32_t)   0, (int32_t)  occa::primitive::not_((int32_t)   1));
  ASSERT_EQ((int64_t)   0, (int64_t)  occa::primitive::not_((int64_t)   1));
}

void testPositive() {
  ASSERT_EQ(true         , (bool)     occa::primitive::positive(true)         );
  ASSERT_EQ((uint8_t)   1, (uint8_t)  occa::primitive::positive((uint8_t)   1));
  ASSERT_EQ((uint16_t)  1, (uint16_t) occa::primitive::positive((uint16_t)  1));
  ASSERT_EQ((uint32_t)  1, (uint32_t) occa::primitive::positive((uint32_t)  1));
  ASSERT_EQ((uint64_t)  1, (uint64_t) occa::primitive::positive((uint64_t)  1));
  ASSERT_EQ((int8_t)    1, (int8_t)   occa::primitive::positive((int8_t)    1));
  ASSERT_EQ((int16_t)   1, (int16_t)  occa::primitive::positive((int16_t)   1));
  ASSERT_EQ((int32_t)   1, (int32_t)  occa::primitive::positive((int32_t)   1));
  ASSERT_EQ((int64_t)   1, (int64_t)  occa::primitive::positive((int64_t)   1));
  ASSERT_EQ((float)     1, (float)    occa::primitive::positive((float)     1));
  ASSERT_EQ((double)    1, (double)   occa::primitive::positive((double)    1));

  ASSERT_EQ(false         , (bool)     occa::primitive::positive(false)        );
  ASSERT_EQ((uint8_t)   -1, (uint8_t)  occa::primitive::positive((uint8_t)  -1));
  ASSERT_EQ((uint16_t)  -1, (uint16_t) occa::primitive::positive((uint16_t) -1));
  ASSERT_EQ((uint32_t)  -1, (uint32_t) occa::primitive::positive((uint32_t) -1));
  ASSERT_EQ((uint64_t)  -1, (uint64_t) occa::primitive::positive((uint64_t) -1));
  ASSERT_EQ((int8_t)    -1, (int8_t)   occa::primitive::positive((int8_t)   -1));
  ASSERT_EQ((int16_t)   -1, (int16_t)  occa::primitive::positive((int16_t)  -1));
  ASSERT_EQ((int32_t)   -1, (int32_t)  occa::primitive::positive((int32_t)  -1));
  ASSERT_EQ((int64_t)   -1, (int64_t)  occa::primitive::positive((int64_t)  -1));
  ASSERT_EQ((float)     -1, (float)    occa::primitive::positive((float)    -1));
  ASSERT_EQ((double)    -1, (double)   occa::primitive::positive((double)   -1));
}

void testNegative() {
  ASSERT_EQ(true          , (bool)     occa::primitive::negative(true)         );
  ASSERT_EQ((uint8_t)   -1, (uint8_t)  occa::primitive::negative((uint8_t)   1));
  ASSERT_EQ((uint16_t)  -1, (uint16_t) occa::primitive::negative((uint16_t)  1));
  ASSERT_EQ((uint32_t)  -1, (uint32_t) occa::primitive::negative((uint32_t)  1));
  ASSERT_EQ((uint64_t)  -1, (uint64_t) occa::primitive::negative((uint64_t)  1));
  ASSERT_EQ((int8_t)    -1, (int8_t)   occa::primitive::negative((int8_t)    1));
  ASSERT_EQ((int16_t)   -1, (int16_t)  occa::primitive::negative((int16_t)   1));
  ASSERT_EQ((int32_t)   -1, (int32_t)  occa::primitive::negative((int32_t)   1));
  ASSERT_EQ((int64_t)   -1, (int64_t)  occa::primitive::negative((int64_t)   1));
  ASSERT_EQ((float)     -1, (float)    occa::primitive::negative((float)     1));
  ASSERT_EQ((double)    -1, (double)   occa::primitive::negative((double)    1));

  ASSERT_EQ(false        , (bool)     occa::primitive::negative(false)        );
  ASSERT_EQ((uint8_t)   1, (uint8_t)  occa::primitive::negative((uint8_t)  -1));
  ASSERT_EQ((uint16_t)  1, (uint16_t) occa::primitive::negative((uint16_t) -1));
  ASSERT_EQ((uint32_t)  1, (uint32_t) occa::primitive::negative((uint32_t) -1));
  ASSERT_EQ((uint64_t)  1, (uint64_t) occa::primitive::negative((uint64_t) -1));
  ASSERT_EQ((int8_t)    1, (int8_t)   occa::primitive::negative((int8_t)   -1));
  ASSERT_EQ((int16_t)   1, (int16_t)  occa::primitive::negative((int16_t)  -1));
  ASSERT_EQ((int32_t)   1, (int32_t)  occa::primitive::negative((int32_t)  -1));
  ASSERT_EQ((int64_t)   1, (int64_t)  occa::primitive::negative((int64_t)  -1));
  ASSERT_EQ((float)     1, (float)    occa::primitive::negative((float)    -1));
  ASSERT_EQ((double)    1, (double)   occa::primitive::negative((double)   -1));
}

void testTilde() {
  ASSERT_EQ(false, (bool) occa::primitive::tilde(true));
  ASSERT_EQ(true , (bool) occa::primitive::tilde(false));
  
  ASSERT_EQ(uint8_t(1), (uint8_t) occa::primitive::tilde(~(uint8_t(1))));
  ASSERT_EQ(int8_t(1), (int8_t)  occa::primitive::tilde(~(int8_t(1))));

  ASSERT_EQ(uint16_t(1), (uint16_t) occa::primitive::tilde(~(uint16_t(1))));
  ASSERT_EQ(int16_t(1), (int16_t)  occa::primitive::tilde(~(int16_t(1))));
  
  ASSERT_EQ(uint32_t(1), (uint32_t) occa::primitive::tilde(~(uint32_t(1))));
  ASSERT_EQ(int32_t(1), (int32_t)  occa::primitive::tilde(~(int32_t(1))));
  
  ASSERT_EQ(uint64_t(1), (uint64_t) occa::primitive::tilde(~(uint64_t(1))));
  ASSERT_EQ(int64_t(1), (int64_t)  occa::primitive::tilde(~(int64_t(1))));

  //Cannot apply tilde to floating point types.
  ASSERT_THROW(occa::primitive::tilde(1.2345f));
  ASSERT_THROW(occa::primitive::tilde(1.2345));
}