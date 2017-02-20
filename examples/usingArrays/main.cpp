#include "occa.hpp"
#include "occa/array.hpp"

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

  occa::array<int> a(3,3);
  occa::array<int, occa::useIdxOrder> b(3,3);

  // b.setIdxOrder("AB", "BA");
  b.setIdxOrder(0,1);

  for (int j = 0; j < (int) a.dim(1); ++j) {
    for (int i = 0; i < (int) a.dim(0); ++i) {
      a(j,i) = 10*j + i;
      b(j,i) = 10*j + i;
    }
  }

  // Can pass @idxOrder to the kernel
  std::cout << "b.idxOrderStr() = " << b.idxOrderStr() << '\n';

  // Basic linear algebra routines
  std::cout << "a.l1Norm()   = " << a.l1Norm<double>() << '\n'
            << "a.l2Norm()   = " << a.l2Norm<double>() << '\n'
            << "a.lpNorm(2)  = " << a.lpNorm<double>(2) << '\n'
            << "a.lInfNorm() = " << a.lInfNorm<double>() << '\n'
            << "a.dot()      = " << a.dot<double>() << '\n'
            << "a.max()      = " << a.max<double>() << '\n'
            << "a.min()      = " << a.min<double>() << '\n';

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

  smallTranspose((int) a.dim(0), a, b);
  occa::finish();

  std::cout << "After:\n";
  printMatrix(a);

  return 0;
}

template <class TM, const int TMi>
void printVector(occa::array<TM,TMi> &a) {
  std::cout << '[';

  for (int i = 0; i < (int) a.entries(); ++i) {
    if (i) std::cout << ", ";
    std::cout << a[i];
  }

  std::cout << "]\n";
}


template <class TM, const int TMi>
void printMatrix(occa::array<TM,TMi> &a) {
  for (int j = 0; j < (int) a.dim(1); ++j) {
    std::cout << "| ";

    for (int i = 0; i < (int) a.dim(0); ++i) {
      if (i) std::cout << ' ';
      std::cout << a(j,i);
    }

    std::cout << " |\n";
  }
}
