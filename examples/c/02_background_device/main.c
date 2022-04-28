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

  // Other useful functions:
  //   occaSetDeviceFromString("mode: 'OpenMP'")
  //   occaDevice = occaGetDevice();
  // Options:
  //   occaSetDeviceFromString("mode: 'Serial'");
  //   occaSetDeviceFromString("mode: 'OpenMP'");
  //   occaSetDeviceFromString("mode: 'CUDA'  , device_id: 0");
  //   occaSetDeviceFromString("mode: 'OpenCL', platform_id: 0, device_id: 0");
  //   occaSetDeviceFromString("mode: 'Metal', device_id: 0");
  //
  // The default device uses "mode: 'Serial'"
  occaSetDeviceFromString(
    occaJsonGetString(
      occaJsonObjectGet(args,
                        "options/device",
                        occaDefault)
    )
  );

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

  // Allocate memory on the background device
  occaMemory o_a  = occaTypedMalloc(entries, occaDtypeFloat, a, occaDefault);
  occaMemory o_b  = occaTypedMalloc(entries, occaDtypeFloat, b, occaDefault);
  occaMemory o_ab = occaTypedMalloc(entries, occaDtypeFloat, ab, occaDefault);

  occaKernel addVectors = occaBuildKernel("addVectors.okl",
                                          "addVectors",
                                          occaDefault);

  occaKernelRun(addVectors,
                occaInt(entries), o_a, o_b, o_ab);
  
  // Copy result to the host
  occaCopyMemToPtr(ab, o_ab, occaAllBytes, 0, occaDefault);

  for (i = 0; i < entries; ++i) {
    printf("%d = %f\n", i, ab[i]);
  }
  for (i = 0; i < entries; ++i) {
    if (fabsf(ab[i] - (a[i] + b[i])) > (2.0f*FLT_EPSILON)) {
      exit(1);
    }
  }

  // Free host memory
  free(a);
  free(b);
  free(ab);

  // Free device memory and occa objects
  occaFree(&args);
  occaFree(&addVectors);
  occaFree(&o_a);
  occaFree(&o_b);
  occaFree(&o_ab);

  return 0;
}

occaJson parseArgs(int argc, const char **argv) {
  occaJson args = occaCliParseArgs(
    argc, argv,
    "{"
    "  description: 'Example showing how to use background devices, allowing passing of the device implicitly',"
    "  options: ["
    "    {"
    "      name: 'device',"
    "      shortname: 'd',"
    "      description: 'Device properties (default: \"{ mode: \\'Serial\\' }\")',"
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
