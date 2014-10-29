#include <iostream>

#include "occa.hpp"

typedef std::vector<occa::device> deviceList_t;

void createLibrary(){
  if(occa::fileExists("testLib"))
    return;

  deviceList_t &devices = occa::getDeviceList();

  const int deviceCount = devices.size();

  for(int i = 0; i < deviceCount; ++i)
    devices[i].cacheKernelInLibrary("addVectors.occa", "addVectors");

  occa::library::save("testLib");
}

void loadFromLibrary(){
  occa::library::load("testLib");
}

int main(int argc, char **argv){
  createLibrary();
  loadFromLibrary();

  occa::kernelDatabase addVectors =
    occa::library::loadKernelDatabase("addVectors");

  deviceList_t &devices     = occa::getDeviceList();
  deviceList_t::iterator it = devices.begin();

  while(it != devices.end()){
    int entries = 5;

    float *a  = new float[entries];
    float *b  = new float[entries];
    float *ab = new float[entries];

    occa::device &device = *(it++);
    occa::memory o_a, o_b, o_ab;

    for(int i = 0; i < entries; ++i){
      a[i]  = i;
      b[i]  = 1 - i;
      ab[i] = 0;
    }

    o_a  = device.malloc(entries*sizeof(float));
    o_b  = device.malloc(entries*sizeof(float));
    o_ab = device.malloc(entries*sizeof(float));

    int dims = 1;
    int itemsPerGroup(2);
    int groups((entries + itemsPerGroup - 1)/itemsPerGroup);

    device[addVectors].setWorkingDims(dims, itemsPerGroup, groups);

    o_a.copyFrom(a);
    o_b.copyFrom(b);

    occa::initTimer(device);

    occa::tic("addVectors");

    addVectors(entries, o_a, o_b, o_ab);

    double elapsedTime = occa::toc("addVectors", addVectors[device]);

    o_ab.copyTo(ab);

    std::cout << "Elapsed time = " << elapsedTime << " s" << std::endl;

    occa::printTimer();

    for(int i = 0; i < 5; ++i)
      std::cout << i << ": " << ab[i] << '\n';

    device[addVectors].free();
    o_a.free();
    o_b.free();
    o_ab.free();

    delete [] a;
    delete [] b;
    delete [] ab;
  }

  return 0;
}
