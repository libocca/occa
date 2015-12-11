#include <iostream>

#include "occa.hpp"
#include "occa/tools.hpp"

int main(int argc, char **argv){
  occa::printAvailableDevices();

  int entries = 37;

  float *a  = new float[entries];
  float *b  = new float[entries];
  float *ab = new float[entries];

  for(int i = 0; i < entries; ++i){
    a[i]  = i;
    b[i]  = 1 - i;
    ab[i] = 0;
  }

  occa::device device;
  occa::kernel addVectors;
  occa::memory o_a, o_b, o_ab;

  device.setup("mode = Serial");

  o_a  = device.malloc(entries*sizeof(float));
  o_b  = device.malloc(entries*sizeof(float));
  o_ab = device.malloc(entries*sizeof(float));

  std::string ispc_file = "addVectors.ispc";
  std::string ispc_flags = " -g -O2 --pic";
  std::string hash = occa::getFileContentHash(ispc_file, ispc_flags);
  std::string ispc_object = occa::sys::getFilename("[ispc]/" + ispc_file + "_" + hash + ".o");
  std::string ispc_sharedobject = occa::sys::getFilename("[ispc]/" + ispc_file + "_" + hash + ".so");

  if(!occa::haveHash(hash)){
    occa::waitForHash(hash);
  }

  if(!occa::sys::fileExists(ispc_object)){
    // Hack to make sure the ispc directory exists.  TODO figure out the occa
    // way to do this?
    if(std::system("mkdir -p ~/._occa/libraries/ispc"))
    {
      occa::releaseHash(hash, 0);
      OCCA_CHECK(false, "problems making ispc dir");
    }

    std::string command = "ispc " + ispc_flags + " " + ispc_file + " -o " + ispc_object;

    std::cout << "Compiling [" << ispc_file << "]\n" << command << "\n";

    if(std::system(command.c_str())){
      occa::releaseHash(hash, 0);
      OCCA_CHECK(false, "ispc compilation error");
    }

    std::string sharedcommand = "g++ -Wl,-E -g -pipe -O2 -pipe -fPIC " +
      ispc_object + " tasksys.cpp -shared -o " + ispc_sharedobject;
    std::cout << "Creating shared object [" << ispc_file << "]\n" << sharedcommand << "\n";

    if(std::system(sharedcommand.c_str())){
      occa::releaseHash(hash, 0);
      OCCA_CHECK(false, "ispc compilation error");
    }
  }

  addVectors = device.buildKernelFromBinary(ispc_sharedobject, "addVectors");
  occa::releaseHash(hash);

  int dims = 1;
  int itemsPerGroup = 16;
  int groups((entries + itemsPerGroup - 1)/itemsPerGroup);

  addVectors.setWorkingDims(dims, itemsPerGroup, groups);

  o_a.copyFrom(a);
  o_b.copyFrom(b);

  addVectors(entries, o_a, o_b, o_ab);

  o_ab.copyTo(ab);

  for(int i = 0; i < entries; ++i)
    std::cout << i << ": " << ab[i] << '\n';

  for(int i = 0; i < entries; ++i){
    if(ab[i] != (a[i] + b[i]))
      throw 1;
  }

  delete [] a;
  delete [] b;
  delete [] ab;

  addVectors.free();
  o_a.free();
  o_b.free();
  o_ab.free();
  device.free();

  return 0;
}
