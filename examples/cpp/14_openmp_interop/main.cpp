#include <iostream>

#include <occa.hpp>

int main(int argc, char **argv) {
  int entries = 5;

  float *a  = new float[entries];
  float *b  = new float[entries];
  float *ab = new float[entries];

  for (int i = 0; i < entries; ++i) {
    a[i]  = i;
    b[i]  = 1 - i;
    ab[i] = 0;
  }


  //---[ OCCA ]-------------------------
  occa::setDevice({
    {"mode", "OpenMP"}
  });

  occa::memory o_a  = occa::wrapMemory<float>(a , entries);
  occa::memory o_b  = occa::wrapMemory<float>(b , entries);
  occa::memory o_ab = occa::wrapMemory<float>(ab, entries);

  occa::kernel addVectors = (
    occa::buildKernel("addVectors.okl",
                       "addVectors")
  );

  addVectors(entries, o_a, o_b, o_ab);
  //====================================

  for (int i = 0; i < entries; ++i) {
    std::cout << i << ": " << ab[i] << '\n';
  }

  delete [] a;
  delete [] b;
  delete [] ab;

  return 0;
}
