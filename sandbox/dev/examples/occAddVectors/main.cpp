#include <iostream>

#include "occa.hpp"

__occa__ void addVectors(const int entries,
                       const float *a,
                       const float *b,
                       float *ab){
  for(int i = 0; i < entries; ++i; tile(16)){
    if(i < entries)
      ab[i] = a[i] + b[i];
  }
}

int main(int argc, char **argv){
  // occa::setDevice("mode = Serial");
  // occa::setDevice("mode = OpenMP  , schedule = compact, chunk = 10");
  // occa::setDevice("mode = OpenCL  , platformID = 0, deviceID = 1");
  // occa::setDevice("mode = CUDA    , deviceID = 0");
  // occa::setDevice("mode = Pthreads, threadCount = 4, schedule = compact, pinnedCores = [0, 0, 1, 1]");
  // occa::setDevice("mode = COI     , deviceID = 0");

  int entries = 5;

  float *a  = occa::new float[entries];
  float *b  = occa::new float[entries];
  float *ab = occa::new float[entries];

  for(int i = 0; i < entries; ++i){
    a[i]  = i;
    b[i]  = 1 - i;
    ab[i] = 0;
  }

  // Call from the host:
  //    addVectors(entries, a, b, ab);

  occa::addVectors(entries, a, b, ab);

  occa::finish();

  for(int i = 0; i < 5; ++i)
    std::cout << i << ": " << ab[i] << '\n';

  for(int i = 0; i < entries; ++i){
    if(ab[i] != (a[i] + b[i]))
      throw 1;
  }

  occa::delete [] a;
  occa::delete [] b;
  occa::delete [] ab;

  return 0;
}
