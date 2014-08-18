#include <iostream>

#include "occa.hpp"


int main(int argc, char **argv){

  int entries = 5;

  float *a  = new float[entries];
  float *b  = new float[entries];
  float *ab = new float[entries];

  for(int i = 0; i < entries; ++i){
    a[i]  = (float) i;
    b[i]  = (float) (1 - i);
    ab[i] = 0;
  }

  int int_size = sizeof(int);
  int pointer_size = sizeof(void*);
  int size_t_size = sizeof(size_t);
  std::cout << "Hello from addVectors: "
      << " integer size: " << int_size
      << " pointer size: " << pointer_size
	  << " size_t size: " << size_t_size << std::endl;
  


  // occa::availableDevices<occa::OpenCL>();
  
  std::string mode = "OpenMP";
  int platformID = 0;
  int deviceID   = 0;
  
  occa::device device;
  occa::kernel addVectors;
  occa::memory o_a, o_b, o_ab;

  device.setup(mode, platformID, deviceID);
  
  o_a  = device.malloc(entries*sizeof(float));
  o_b  = device.malloc(entries*sizeof(float));
  o_ab = device.malloc(entries*sizeof(float));

  char *occaDir_ = getenv("OCCA_DIR");
  std::string addVectors_occa("addVectors.occa");
  if(occaDir_ != NULL) {
	  std::string occaDir(occaDir_);
	  addVectors_occa = occaDir + "/examples/addVectors/" + addVectors_occa;
  }

  addVectors = device.buildKernelFromSource(addVectors_occa.c_str(),
                                            "addVectors");

  int dims = 1;
  int itemsPerGroup(2);
  int groups((entries + itemsPerGroup - 1)/itemsPerGroup);

  addVectors.setWorkingDims(dims, itemsPerGroup, groups);

  o_a.copyFrom(a);
  o_b.copyFrom(b);

  addVectors(entries, o_a, o_b, o_ab);

  o_ab.copyTo(ab);

  for(int i = 0; i < 5; ++i)
    std::cout << i << ": " << ab[i] << '\n';

  delete [] a;
  delete [] b;
  delete [] ab;

  addVectors.free();
  o_a.free();
  o_b.free();
  o_ab.free();
  device.free();
  
}
