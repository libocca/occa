#include <vector>

#include <occa/internal/utils/testing.hpp>

void testAsserts();
void testFailedAsserts();

int main(const int argc, const char **argv) {
  testAsserts();
  testFailedAsserts();

  return 0;
}

void testAsserts() {
  ASSERT_LE(1, 1);
  ASSERT_LT(1, 2);
  ASSERT_GE(2, 2);
  ASSERT_GT(2, 1);
  ASSERT_EQ(1, 1);
  ASSERT_NEQ(1, 2);

  // float / double
  ASSERT_EQ((float) 1, (float) (1 + 1e-9));
  ASSERT_NEQ((float) 1, (float) (1 + 1e-7));

  ASSERT_EQ((double) 1, (float) (1 + 1e-9));
  ASSERT_NEQ((double) 1, (float) (1 + 1e-7));

  ASSERT_EQ((float) 1, (double) (1 + 1e-9));
  ASSERT_NEQ((float) 1, (double) (1 + 1e-7));

  ASSERT_EQ((double) 1, (double) (1 + 1e-15));
  ASSERT_NEQ((double) 1, (double) (1 + 1e-13));

  ASSERT_EQ_BINARY(1, 1);
  ASSERT_NEQ_BINARY(1, 2);

  ASSERT_TRUE(1 == 1);
  ASSERT_FALSE(1 == 2);

  std::vector<int> vec;
  vec.push_back(1);
  ASSERT_IN(1, vec);
  ASSERT_NOT_IN(2, vec);
}

void testFailedAsserts() {
  ASSERT_THROW({
    ASSERT_LE(2, 1);
  });

  ASSERT_THROW({
    ASSERT_LT(2, 1);
  });

  ASSERT_THROW({
    ASSERT_GE(1, 2);
  });

  ASSERT_THROW({
    ASSERT_GT(1, 2);
  });

  ASSERT_THROW({
    ASSERT_EQ(1, 2);
  });

  ASSERT_THROW({
    ASSERT_NEQ(1, 1);
  });

  ASSERT_THROW({
    ASSERT_EQ_BINARY(1, 2);
  });

  ASSERT_THROW({
    ASSERT_NEQ_BINARY(1, 1);
  });

  ASSERT_THROW({
    ASSERT_TRUE(1 == 2);
  });

  ASSERT_THROW({
    ASSERT_FALSE(1 == 1);
  });

  std::vector<int> vec;
  vec.push_back(1);

  ASSERT_THROW(
    ASSERT_IN(2, vec);
  );

  ASSERT_THROW(
    ASSERT_NOT_IN(1, vec);
  );
}
