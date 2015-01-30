#include <iostream>

#include "occa.hpp"

int main(int argc, char **argv){


  const int width  = 16;
  const int height = 16;

  const int sz = width*height*sizeof(float);

  float *A	= new float[height*width];
  float *transA = new float[height*width];

  for(int i=0; i<height*width; i++)
    A[i] = i;

  std::string mode = "OpenCL";
  int platformID = 0;
  int deviceID = 2;

  occa::device device;
  occa::memory o_A, o_transA;

  device.setup(mode, platformID, deviceID);

  const int dim = 2;

  o_A = device.textureAlloc(dim, occa::dim(width, height),
                      A,
                      occa::floatFormat, occa::readOnly);

  A[0] = 100;

  o_A.copyTo(A);

  if(A[0] == 100){
    std::cout << "Failed.\n";
    return 0;
  }

  o_transA = device.textureAlloc(dim, occa::dim(height, width),
                           transA,
                           occa::floatFormat, occa::readWrite);

  occa::kernel textureTranspose
    = device.buildKernelFromSource("textureTranspose.occa",
				   "textureTranspose");

  const int BX = 16;
  const int BY = 16;

  size_t dims = 2;
  occa::dim inner(BX, BY);
  occa::dim outer((width+BX-1)/BX, (height+BY-1)/BY);

  textureTranspose.setWorkingDims(dims, inner, outer);

  textureTranspose(width, height, o_A, o_transA);

  // copy back to host and verify results
  o_transA.copyTo(transA);

  for(int i=0; i<width; i++){
    for(int j=0; j<height; j++){

      if(A[j*width+i] != transA[i*height+j])
        std::cout << A[j*width + i] << " != " << transA[i*height+j] << '\n';

    }
  }

  delete [] A;
  delete [] transA;

  textureTranspose.free();
  o_A.free();
  o_transA.free();
  device.free();
}
