#include "occa.hpp"
#include "occa/array.hpp"

int main(int argc, char **argv){
  occa::array<double> a(3,3);
  occa::array<double, occa::useIdxOrder> b(3,3);

  b.setIdxOrder("AB", "BA");
  // b.setIdxOrder(0,1);

  for(int j = 0; j < (int) a.dim(1); ++j){
    for(int i = 0; i < (int) a.dim(0); ++i){
      a(j,i) = 10*j + i;
      b(j,i) = 10*j + i;
    }
  }

  std::cout << "b.idxOrderStr() = " << b.idxOrderStr() << '\n';

  for(int i = 0; i < (int) a.entries(); ++i){
    std::cout << "a[i] = " << a[i] << '\n'
              << "b[i] = " << b[i] << '\n';
  }

  return 0;
}
