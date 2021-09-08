#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <occa.h>

//---[ Internal Tools ]-----------------
// Note: These headers are not officially supported
//       Please don't rely on it outside of the occa examples
#include <occa/internal/c/cli.h>
//======================================

occaJson parseArgs(int argc, const char **argv);

int main(int argc, const char **argv) {
  occaJson args = parseArgs(argc, argv);

  // occaSetDeviceFromString("mode: 'Serial'");
  // occaSetDeviceFromString("mode: 'OpenMP'");
  // occaSetDeviceFromString("mode: 'CUDA'  , device_id: 0");
  // occaSetDeviceFromString("mode: 'OpenCL', platform_id: 0, device_id: 0");
  // occaSetDeviceFromString("mode: 'Metal', device_id: 0");
  occaSetDeviceFromString(
    occaJsonGetString(
      occaJsonObjectGet(args,
                        "options/device",
                        occaDefault)
    )
  );

  // Choosing something not divisible by 256
  int i;
  int entries = 10000;
  int block   = 256;
  int blocks  = (entries + block - 1)/block;

  float *vec      = (float*) malloc(entries * sizeof(float));
  float *blockSum = (float*) malloc(blocks  * sizeof(float));

  float sum = 0;

  // Initialize device memory
  for (i = 0; i < entries; ++i) {
    vec[i] = 1;
    sum += vec[i];
  }

  for (i = 0; i < blocks; ++i) {
    blockSum[i] = 0;
  }

  // Allocate memory on the device
  occaMemory o_vec      = occaTypedMalloc(entries, occaDtypeFloat, NULL, occaDefault);
  occaMemory o_blockSum = occaTypedMalloc(blocks , occaDtypeFloat, NULL, occaDefault);

  // Pass value of 'block' at kernel compile-time
  occaJson reductionProps = occaCreateJson();
  occaJsonObjectSet(reductionProps,
                    "defines/block",
                    occaInt(block));

  occaKernel reduction = occaBuildKernel("reduction.okl",
                                         "reduction",
                                         reductionProps);

  // Host -> Device
  occaCopyPtrToMem(o_vec, vec,
                   occaAllBytes, 0, occaDefault);

  occaKernelRun(reduction,
                occaInt(entries), o_vec, o_blockSum);

  // Host <- Device
  occaCopyMemToPtr(blockSum, o_blockSum,
                   occaAllBytes, 0, occaDefault);

  // Finalize the reduction in the host
  for (i = 1; i < blocks; ++i) {
    blockSum[0] += blockSum[i];
  }

  // Validate
  if (fabs(blockSum[0] - sum) > 1.0e-8) {
    printf("sum      = %f\n", sum);
    printf("blockSum = %f\n", blockSum[0]);
    printf("Reduction failed\n");
    exit(1);
  }
  else {
    printf("Reduction = %f\n", blockSum[0]);
  }

  // Free host memory
  free(vec);
  free(blockSum);

  occaFree(&args);
  occaFree(&reductionProps);
  occaFree(&reduction);
  occaFree(&o_vec);
  occaFree(&o_blockSum);

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
