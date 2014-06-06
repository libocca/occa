#include "stdlib.h"
#include "stdio.h"

#include "occa_c.hpp"

int main(int argc, char **argv){
  int entries = 5;
  int i;

  float *a  = (float*) malloc(entries*sizeof(float));
  float *b  = (float*) malloc(entries*sizeof(float));
  float *ab = (float*) malloc(entries*sizeof(float));

  for(i = 0; i < entries; ++i){
    a[i]  = i;
    b[i]  = 1 - i;
    ab[i] = 0;
  }

  const char *mode = "OpenMP";
  int platformID = 0;
  int deviceID   = 0;

  occaDevice device;
  occaKernel addVectors;
  occaMemory o_a, o_b, o_ab;

  device = occaGetDevice(mode, platformID, deviceID);

  o_a  = device.malloc(entries*sizeof(float));
  o_b  = device.malloc(entries*sizeof(float));
  o_ab = device.malloc(entries*sizeof(float));

  addVectors = occaBuildKernelFromSource(device,
                                         "addVectors.occa", "addVectors",
                                         occaNoKernelInfo);

  int dims = 1;
  occaDim itemsPerGroup, groups;

  itemsPerGroup.x = 2;
   group.x        = (entries + itemsPerGroup - 1)/itemsPerGroup;

  occaKernelSetWorkingDims(addVectors,
                           dims, itemsPerGroup, groups);

  occaCopyFromPtr(o_a, a, entries*sizeof(float), 0);
  occaCopyFromPtr(o_b, b, occaAutoSize, occaNoOffset);

  occaRunKernel(addVectors,
                occaInt(entries),
                o_a, o_b, o_ab);

  occaCopyToPtr(ab, o_ab, occaAutoSize, occaNoOffset);

  for(i = 0; i < 5; ++i)
    printf("%d = %f\n", i, ab[i]);

  free(a);
  free(b);
  free(ab);

  occaFreeKernel(addVectors);
  occaFreeMemory(o_a);
  occaFreeMemory(o_b);
  occaFreeMemory(o_ab);
  occaFreeDevice(device);
}



