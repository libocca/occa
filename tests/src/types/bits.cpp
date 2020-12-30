#include <occa/internal/utils/testing.hpp>

#include <occa.hpp>

int main(const int argc, const char **argv) {
  occa::bitfield bf1(1 << 0);
  ASSERT_EQ_BINARY(bf1.b1, 0UL);
  ASSERT_EQ_BINARY(bf1.b2, 1UL);

  bf1 <<= (occa::bitfield::bits() / 2);
  ASSERT_EQ_BINARY(bf1.b1, 1UL);
  ASSERT_EQ_BINARY(bf1.b2, 0UL);

  bf1 >>= (occa::bitfield::bits() / 2);
  ASSERT_EQ_BINARY(bf1.b1, 0UL);
  ASSERT_EQ_BINARY(bf1.b2, 1UL);

  occa::bitfield bf2 = (
    occa::bitfield(1 << 0) |
    occa::bitfield(1 << 1)
  );

  ASSERT_TRUE(bf1 & bf2);
  bf2 <<= 1;
  ASSERT_FALSE(bf1 & bf2);

  const occa::bitfield a1(1L << 0, 1L << 0);
  const occa::bitfield a2(1L << 1, 1L << 1);
  const occa::bitfield b1(1L << 2, 1L << 2);
  const occa::bitfield b2(1L << 3, 1L << 3);
  const occa::bitfield c1(1L << 4, 1L << 4);
  const occa::bitfield c2(1L << 5, 1L << 5);

  const occa::bitfield a = (a1 | a2);
  const occa::bitfield b = (b1 | b2);
  const occa::bitfield c = (c1 | c2);

  const occa::bitfield start = (a1 | b1 | c1);
  const occa::bitfield end   = (a2 | b2 | c2);

  ASSERT_TRUE(a & a1);
  ASSERT_TRUE(a & a2);

  ASSERT_TRUE(start & a);
  ASSERT_TRUE(start & a1);
  ASSERT_TRUE(start & b1);
  ASSERT_TRUE(start & c1);

  ASSERT_TRUE(end & a);
  ASSERT_TRUE(end & a2);
  ASSERT_TRUE(end & b2);
  ASSERT_TRUE(end & c2);

  ASSERT_FALSE(a & b);
  ASSERT_FALSE(a & c);
  ASSERT_FALSE(b & c);

  ASSERT_FALSE(start & end);

  ASSERT_TRUE(a1 != a2);
  ASSERT_TRUE(a1 <  a2);
  ASSERT_TRUE(a2 <= a2);
  ASSERT_TRUE(a2 == a2);
  ASSERT_TRUE(a2 >= a2);
  ASSERT_TRUE(a2 >  a1);

  // Test bad shifts
  occa::bitfield e(1L << 0);
  occa::bitfield e1 = (e << 0);
  occa::bitfield e2 = (e << -1);

  ASSERT_EQ(e, e1);
  ASSERT_EQ(e, e2);
  ASSERT_EQ(e1, e2);

  e1 = (e >> 0);
  e2 = (e >> -1);

  ASSERT_EQ(e, e1);
  ASSERT_EQ(e, e2);
  ASSERT_EQ(e1, e2);

  // Test too much shift
  e1 = e << 1000;
  e2 = e >> 1000;
  ASSERT_FALSE(e1);
  ASSERT_FALSE(e2);
}
