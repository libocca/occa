#include "stdlib.h"
#include "stdio.h"

#include "occa_c.h"

int main(int argc, char **argv){
  occaPrintAvailableDevices();

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

  occaDevice device;
  occaKernel addVectors;
  occaMemory o_a, o_b, o_ab;

  //---[ Device setup with string flags ]-------------------
  const char *deviceInfo = "mode = OpenCL  , platformID = 0, deviceID = 0";

  device = occaCreateDevice(deviceInfo);

  o_a  = occaDeviceMalloc(device, entries*sizeof(float), NULL);
  o_b  = occaDeviceMalloc(device, entries*sizeof(float), NULL);
  o_ab = occaDeviceMalloc(device, entries*sizeof(float), NULL);

  occaKernelInfo info = occaCreateKernelInfo();
  occaKernelInfoAddDefine(info, "DIMENSION", occaInt(10));

  addVectors =  occaDeviceBuildKernelFromFloopy(device,
                                     "addVectors.floopy", "addVectors",
                                     info);

  /* addVectors = occaBuildKernelFromSource(device, */
  /*                                        "addVectors.occa", "addVectors", */
  /*                                        occaNoKernelInfo); */

  int dims = 1;
  occaDim itemsPerGroup, groups;

  itemsPerGroup.x = 2;
  groups.x        = (entries + itemsPerGroup.x - 1)/itemsPerGroup.x;

  occaKernelSetWorkingDims(addVectors,
                           dims, itemsPerGroup, groups);

  occaCopyPtrToMem(o_a, a, entries*sizeof(float), 0);
  occaCopyPtrToMem(o_b, b, occaAutoSize, occaNoOffset);

  occaKernelRun(addVectors,
               occaInt(entries), o_a, o_b, o_ab);

  occaCopyMemToPtr(ab, o_ab, occaAutoSize, occaNoOffset);

  for(i = 0; i < 5; ++i)
    printf("%d = %f\n", i, ab[i]);

  for(i = 0; i < entries; ++i){
    if(ab[i] != (a[i] + b[i]))
      exit(1);
  }

  free(a);
  free(b);
  free(ab);

  occaKernelInfoFree(info);
  occaKernelFree(addVectors);
  occaMemoryFree(o_a);
  occaMemoryFree(o_b);
  occaMemoryFree(o_ab);
  occaDeviceFree(device);
}
