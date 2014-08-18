#include "stdlib.h"
#include "stdio.h"

#include "occa_c.h"

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

  const char *mode = "Pthreads";
  int platformID = 0;
  int deviceID   = 0;

  occaDevice device;
  occaKernel addVectors;
  occaMemory o_a, o_b, o_ab;

  device = occaGetDevice(mode, platformID, deviceID);

  o_a  = occaDeviceMalloc(device, entries*sizeof(float), NULL);
  o_b  = occaDeviceMalloc(device, entries*sizeof(float), NULL);
  o_ab = occaDeviceMalloc(device, entries*sizeof(float), NULL);

  addVectors = occaBuildKernelFromSource(device,
                                         "addVectors.occa", "addVectors",
                                         occaNoKernelInfo);

  int dims = 1;
  occaDim itemsPerGroup, groups;

  itemsPerGroup.x = 2;
  groups.x        = (entries + itemsPerGroup.x - 1)/itemsPerGroup.x;

  occaKernelSetWorkingDims(addVectors,
                           dims, itemsPerGroup, groups);

  occaCopyPtrToMem(o_a, a, entries*sizeof(float), 0);
  occaCopyPtrToMem(o_b, b, occaAutoSize, occaNoOffset);

  /* occaArgumentList list = occaGenArgumentList(); */

  /* occaArgumentListAddArg(list, 0, occaInt(entries)); */
  /* occaArgumentListAddArg(list, 1, o_a); */
  /* occaArgumentListAddArg(list, 2, o_b); */
  /* occaArgumentListAddArg(list, 3, o_ab); */

  /* occaKernelRun_(addVectors, list); */

  occaKernelRun(addVectors,
               occaInt(entries), o_a, o_b, o_ab);

  /* occaArgumentListClear(list); */
  /* occaArgumentListFree(list); */

  occaCopyMemToPtr(ab, o_ab, occaAutoSize, occaNoOffset);

  for(i = 0; i < 5; ++i)
    printf("%d = %f\n", i, ab[i]);

  free(a);
  free(b);
  free(ab);

  occaKernelFree(addVectors);
  occaMemoryFree(o_a);
  occaMemoryFree(o_b);
  occaMemoryFree(o_ab);
  occaDeviceFree(device);
}
