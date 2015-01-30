#include <iostream>

#include "occa.hpp"

int main(int argc, char **argv){
  const int width  = 10000;

  const int sz = width*sizeof(float);

  float *A	= new float[width];
  float *copyA = new float[width];

  for(int i=0; i<width; i++)
    A[i] = i;

  std::vector<int> perm(width, 0);

  for(int i=0; i<width; i++)
    perm[i] = width-i-1;

  std::string mode = "OpenCL";
  int platformID = 0;
  int deviceID   = 0;

  occa::device device;
  occa::memory o_A, o_copyA;

  device.setup(mode, platformID, deviceID);

  const int dim = 1;

  o_A = device.textureAlloc(dim, occa::dim(width),
                      A,
                      occa::floatFormat, occa::readWrite);

  A[0] = 100;

  o_A.copyTo(A);

  if(A[0] == 100){
    std::cout << "Failed.\n";
    return 0;
  }

  o_copyA = device.textureAlloc(dim, occa::dim(width),
			  copyA,
			  occa::floatFormat, occa::readWrite);

  occa::memory o_perm = device.malloc(width*sizeof(int), &(perm[0]));

  occa::kernel copyKernel
    = device.buildKernelFromSource("copy.occa",
				   "copy");

  const int BX = 16;

  size_t dims = 1;
  occa::dim inner(BX);
  occa::dim outer((width+BX-1)/BX);

  copyKernel.setWorkingDims(dims, inner, outer);

  copyKernel(width, o_perm, o_A, o_copyA);

  // copy back to host and verify results
  o_copyA.copyTo(copyA);

  std::cout
    << "copyA[0] = " << copyA[0] << '\n';

  throw 1;

  for(int i=0; i<width; i++){
    if(A[i]+2 != copyA[perm[i]])
      std::cout << A[i]+2 << " != " << copyA[perm[i]] << '\n';
  }

  delete [] A;
  delete [] copyA;

  copyKernel.free();
  o_A.free();
  o_copyA.free();
  device.free();
}
