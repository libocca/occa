#include "occa.hpp"

template <class TM, const int TMi>
void printVector(occa::array<TM,TMi> &a);

template <class TM, const int TMi>
void printMatrix(occa::array<TM,TMi> &a);

int main(int argc, char **argv) {
  // occa::setDevice("mode       : 'OpenCL', "
  //                 "platformID : 0, "
  //                 "deviceID   : 1, ");

  // occa::setDevice("mode     : 'CUDA', "
  //                 "deviceID : 0, ");

  //---[ Testing API ]------------------
  std::cout << "Testing API:\n";

  occa::array<float> vec1(10), vec2(10);
  for (int i = 0; i < 10; ++i) {
    vec1[i] = vec2[i] = i + 1;
  }

  printVector(vec1);
  vec1 += 100;
  std::cout << "vec1 += 100\n";
  printVector(vec1);
  vec1 -= 100;
  std::cout << "vec1 -= 100\n";
  printVector(vec1);
  vec1 *= 10;
  std::cout << "vec1 *= 10\n";
  printVector(vec1);
  vec1 /= 10;
  std::cout << "vec1 /= 10\n";
  printVector(vec1);

  vec1 += vec2;
  std::cout << "vec1 += vec2\n";
  printVector(vec1);
  vec1 -= vec2;
  std::cout << "vec1 -= vec2\n";
  printVector(vec1);
  vec1 *= vec2;
  std::cout << "vec1 *= vec2\n";
  printVector(vec1);
  vec1 /= vec2;
  std::cout << "vec1 /= vec2\n";
  printVector(vec1);

  std::cout << "vec1 += -1*vec2 (axpy)\n";
  vec1.sum(-1, vec2);
  printVector(vec1);

  std::cout << "vec1 += 10*vec2 (axpy)\n";
  vec1.sum(10, vec2);
  printVector(vec1);

  occa::array<float> subvec1 = vec1 + 5;
  std::cout << "vec1 + 5\n";
  printVector(subvec1);

  // Basic linear algebra routines
  std::cout << "vec1.l1Norm()       = " << vec1.l1Norm<double>() << '\n'
            << "vec1.l2Norm()       = " << vec1.l2Norm<double>() << '\n'
            << "vec1.lpNorm(2)      = " << vec1.lpNorm<double>(2) << '\n'
            << "vec1.lInfNorm()     = " << vec1.lInfNorm<double>() << '\n'
            << "vec1.dot(vec2)      = " << vec1.dot<double>(vec2) << '\n'
            << "vec1.distance(vec2) = " << vec1.distance<double>(vec2) << '\n'
            << "vec1.max()          = " << vec1.max() << '\n'
            << "vec1.min()          = " << vec1.min() << '\n';

  occa::array<int> a(3,3);
  occa::array<int, occa::dynamic> b(3,3);
  b.reindex(1,0);

  for (int j = 0; j < (int) a.dim(1); ++j) {
    for (int i = 0; i < (int) a.dim(0); ++i) {
      a(i,j) = 10*j + i;
      b(i,j) = 10*j + i;
    }
  }

  // Can pass @idxOrder to the kernel
  std::cout << "b.indexingStr() = " << b.indexingStr() << '\n';

  // Arrays a and b print out differently
  //   due to b.setIdxOrder()
  printVector(a);
  printVector(b);

  //---[ Testing Kernel ]---------------
  std::cout << "Testing Kernel:\n";

  occa::kernel smallTranspose = occa::buildKernel("smallTranspose.okl",
                                                  "smallTranspose");

  std::cout << "Before:\n";
  printMatrix(a);

  smallTranspose((int) a.dim(0),
                 a.arrayArg(),
                 b.arrayArg());
  occa::finish();

  std::cout << "After:\n";
  printMatrix(a);

  return 0;
}

template <class TM, const int TMi>
void printVector(occa::array<TM,TMi> &a) {
  occa::finish();
  std::cout << '[';

  for (int i = 0; i < (int) a.size(); ++i) {
    if (i) std::cout << ", ";
    std::cout << a[i];
  }

  std::cout << "]\n";
}


template <class TM, const int TMi>
void printMatrix(occa::array<TM,TMi> &a) {
  occa::finish();
  for (int j = 0; j < (int) a.dim(1); ++j) {
    std::cout << "| ";

    for (int i = 0; i < (int) a.dim(0); ++i) {
      if (i) std::cout << ' ';
      std::cout << a(j,i);
    }

    std::cout << " |\n";
  }
}
