#include <occa/internal/utils/testing.hpp>

#include <occa.hpp>

int main(const int argc, const char **argv) {
#define TEST_DIM(dims_, x_, y_, z_)             \
  ASSERT_EQ(dim.dims, dims_);                   \
  ASSERT_EQ(dim.x, (occa::udim_t) x_);          \
  ASSERT_EQ(dim.y, (occa::udim_t) y_);          \
  ASSERT_EQ(dim.z, (occa::udim_t) z_)

  // Constructors
  occa::dim dim;
  TEST_DIM(0, 1, 1, 1);

  dim = occa::dim(10);
  TEST_DIM(1, 10, 1, 1);

  dim = occa::dim(10, 20);
  TEST_DIM(2, 10, 20, 1);

  dim = occa::dim(10, 20, 30);
  TEST_DIM(3, 10, 20, 30);

  dim = occa::dim(1, 10, 20, 30);
  TEST_DIM(1, 10, 20, 30);

  dim = occa::dim(1, 10, 20, 30);
  TEST_DIM(1, 10, 20, 30);
  ASSERT_NEQ(dim,
             occa::dim(10, 20, 30));
  ASSERT_EQ(dim,
            occa::dim(1, 10, 20, 30));

  dim = dim - occa::dim(10, 10, 10);
  TEST_DIM(3, 0, 10, 20);

  dim = dim + occa::dim(10, 10, 10);
  TEST_DIM(3, 10, 20, 30);

  dim = dim * occa::dim(10, 10, 10);
  TEST_DIM(3, 100, 200, 300);

  dim = dim / occa::dim(10, 10, 10);
  TEST_DIM(3, 10, 20, 30);
  ASSERT_EQ(dim,
            occa::dim(10, 20, 30));

  // [] indexing
  ASSERT_EQ(dim[0],
            (occa::udim_t) 10);
  ASSERT_EQ(dim[1],
            (occa::udim_t) 20);
  ASSERT_EQ(dim[2],
            (occa::udim_t) 30);

  const occa::dim constDim(10, 20, 30);
  ASSERT_EQ(constDim[0],
            (occa::udim_t) 10);
  ASSERT_EQ(constDim[1],
            (occa::udim_t) 20);
  ASSERT_EQ(constDim[2],
            (occa::udim_t) 30);

  std::cout << constDim << '\n';

#undef TEST_DIM

  return 0;
}
