#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include <occa.h>

//---[ Internal Tools ]-----------------
// Note: These headers are not officially supported
//       Please don't rely on it outside of the occa examples
#include <occa/internal/c/cli.h>
//======================================

occaJson parseArgs(int argc, const char **argv);

int main(int argc, const char **argv) {
  occaJson args = parseArgs(argc, argv);

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
  device = occaCreateDeviceFromString(
    occaJsonGetString(
      occaJsonObjectGet(args,
                        "options/device",
                        occaDefault)
    )
  );

  // device = occaCreateDeviceFromString(
  //   "mode: 'Serial'"
  // );

  // device = occaCreateDeviceFromString(
  //   "mode     : 'OpenMP',"
  //   "schedule : 'compact',"
  //   "chunk    : 10"
  // );

  // device = occaCreateDeviceFromString(
  //   "mode      : 'CUDA',"
  //   "device_id : 0"
  // );

  // device = occaCreateDeviceFromString(
  //   "mode        : 'OpenCL',"
  //   "platform_id : 0,"
  //   "device_id   : 1"
  // );

  // device = occaCreateDeviceFromString(
  //   "mode        : 'Metal', "
  //   "device_id   : 1"
  // );
  //========================================================

  // Allocate memory on the device
  o_a  = occaDeviceTypedMalloc(device, entries, occaDtypeFloat, a, occaDefault);
  o_b  = occaDeviceTypedMalloc(device, entries, occaDtypeFloat, b, occaDefault);

  // We can also allocate memory without a dtype and manually copy the data over
  // WARNING: This will disable runtime type checking
  o_ab = occaDeviceMalloc(device, entries * sizeof(float), NULL, occaDefault);
  occaCopyPtrToMem(o_ab, ab, occaAllBytes, 0, occaDefault);

  // Setup properties that can be passed to the kernel
  occaJson props = occaCreateJson();
  occaJsonObjectSet(props, "defines/TILE_SIZE", occaInt(10));

  // Compile the kernel at run-time
  addVectors = occaDeviceBuildKernel(device,
                                     "addVectors.okl",
                                     "addVectors",
                                     props);

  // Launch device kernel
  occaKernelRun(addVectors,
                occaInt(entries), o_a, o_b, o_ab);

  // Copy result to the host
  occaCopyMemToPtr(ab, o_ab, occaAllBytes, 0, occaDefault);

  // Assert values
  for (i = 0; i < entries; ++i) {
    printf("%d = %f\n", i, ab[i]);
  }
  for (i = 0; i < entries; ++i) {
    if (fabsf(ab[i] -(a[i] + b[i])) > (2.0f*FLT_EPSILON))
      exit(1);
  }

  // Free host memory
  free(a);
  free(b);
  free(ab);

  // Free device memory and occa objects
  occaFree(&args);
  occaFree(&props);
  occaFree(&addVectors);
  occaFree(&o_a);
  occaFree(&o_b);
  occaFree(&o_ab);
  occaFree(&device);
}

occaJson parseArgs(int argc, const char **argv) {
  occaJson args = occaCliParseArgs(
    argc, argv,
    "{"
    "  description: 'Example adding two vectors',"
    "  options: ["
    "    {"
    "      name: 'device',"
    "      shortname: 'd',"
    "      description: 'Device properties (default: \"{mode: \\'Serial\\'}\")',"
    "      with_arg: true,"
    "      default_value: \"{ mode: 'Serial' }\","
    "    },"
    "    {"
    "      name: 'verbose',"
    "      shortname: 'v',"
    "      description: 'Compile kernels in verbose mode',"
    "      default_value: false,"
    "    },"
    "  ],"
    "}"
  );

  occaJson settings = occaSettings();
  occaJsonObjectSet(settings,
                    "kernel/verbose",
                    occaJsonObjectGet(args, "options/verbose", occaBool(0)));

  return args;
}
