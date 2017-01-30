#include "stdlib.h"
#include "stdio.h"

#include "occa_c.h"

int main(int argc, char **argv) {
  occaPrintModeInfo();

  int entries = 5;
  int i;

  float *a  = (float*) malloc(entries*sizeof(float));
  float *b  = (float*) malloc(entries*sizeof(float));
  float *ab = (float*) malloc(entries*sizeof(float));

  for (i = 0; i < entries; ++i) {
    a[i]  = i;
    b[i]  = 1 - i;
    ab[i] = 0;
  }

  occaDevice device;
  occaKernel addVectors;
  occaMemory o_a, o_b, o_ab;

  //---[ Device setup with string flags ]-------------------
  const char *deviceInfo = "mode: 'Serial'";

  /*
  const char *deviceInfo = ("mode     : 'OpenMP', "
                            "schedule : 'compact', "
                            "chunk    : 10");

  const char *deviceInfo = ("mode       : 'OpenCL', "
                            "platformID : 0, "
                            "deviceID   : 1");

  const char *deviceInfo = ("mode     : 'CUDA', "
                            "deviceID : 0");

  const char *deviceInfo = ("mode        : 'Threads', "
                            "threadCount : 4, "
                            "schedule    : 'compact', "
                            "pinnedCores : [0, 0, 1, 1]");
  */
  device = occaCreateDevice(occaString(deviceInfo));

  o_a  = occaDeviceMalloc(device, entries*sizeof(float), NULL, occaDefault);
  o_b  = occaDeviceMalloc(device, entries*sizeof(float), NULL, occaDefault);
  o_ab = occaDeviceMalloc(device, entries*sizeof(float), NULL, occaDefault);

  occaProperties props = occaCreateProperties();
  occaPropertiesSet(props, "DIMENSION", occaInt(10));

  addVectors = occaDeviceBuildKernel(device,
                                     "addVectors.okl", "addVectors",
                                     props);

  occaCopyPtrToMem(o_a, a, entries*sizeof(float), 0, occaDefault);
  occaCopyPtrToMem(o_b, b, occaAllBytes         , 0, occaDefault);

  occaKernelRun(addVectors,
                occaInt(entries), o_a, o_b, o_ab);

  occaCopyMemToPtr(ab, o_ab, occaAllBytes, 0, occaDefault);

  for (i = 0; i < 5; ++i) {
    printf("%d = %f\n", i, ab[i]);
  }
  for (i = 0; i < entries; ++i) {
    if (ab[i] != (a[i] + b[i]))
      exit(1);
  }

  free(a);
  free(b);
  free(ab);

  occaPropertiesFree(props);
  occaKernelFree(addVectors);
  occaMemoryFree(o_a);
  occaMemoryFree(o_b);
  occaMemoryFree(o_ab);
  occaDeviceFree(device);
}
