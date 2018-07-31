/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 David Medina and Tim Warburton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */
#include <vector>

#include <occa/tools/testing.hpp>

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
  ASSERT_THROW_START {
    ASSERT_LE(2, 1);
  } ASSERT_THROW_END;

  ASSERT_THROW_START {
    ASSERT_LT(2, 1);
  } ASSERT_THROW_END;

  ASSERT_THROW_START {
    ASSERT_GE(1, 2);
  } ASSERT_THROW_END;

  ASSERT_THROW_START {
    ASSERT_GT(1, 2);
  } ASSERT_THROW_END;

  ASSERT_THROW_START {
    ASSERT_EQ(1, 2);
  } ASSERT_THROW_END;

  ASSERT_THROW_START {
    ASSERT_NEQ(1, 1);
  } ASSERT_THROW_END;

  ASSERT_THROW_START {
    ASSERT_EQ_BINARY(1, 2);
  } ASSERT_THROW_END;

  ASSERT_THROW_START {
    ASSERT_NEQ_BINARY(1, 1);
  } ASSERT_THROW_END;

  ASSERT_THROW_START {
    ASSERT_TRUE(1 == 2);
  } ASSERT_THROW_END;

  ASSERT_THROW_START {
    ASSERT_FALSE(1 == 1);
  } ASSERT_THROW_END;

  std::vector<int> vec;
  vec.push_back(1);

  ASSERT_THROW(
    ASSERT_IN(2, vec);
  );

  ASSERT_THROW(
    ASSERT_NOT_IN(1, vec);
  );
}
