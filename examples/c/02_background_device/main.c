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

  float *a  = (float*) occaTypedUMalloc(entries, occaDtypeFloat, NULL, occaDefault);
  float *b  = (float*) occaTypedUMalloc(entries, occaDtypeFloat, NULL, occaDefault);
  float *ab = (float*) occaTypedUMalloc(entries, occaDtypeFloat, NULL, occaDefault);

  for (i = 0; i < entries; ++i) {
    a[i]  = i;
    b[i]  = 1 - i;
    ab[i] = 0;
  }

  occaKernel addVectors = occaBuildKernel("addVectors.okl",
                                          "addVectors",
                                          occaDefault);

  // Arrays a, b, and ab are now resident
  //   on [device]
  occaKernelRun(addVectors,
                occaInt(entries), occaPtr(a), occaPtr(b), occaPtr(ab));

  // b is not const in the kernel, so we can use
  //   dontSync(b) to manually force b to not sync
  occaDontSync(b);

  // Finish work queued up in [device],
  //   synchronizing a, b, and ab and
  //   making it safe to use them again
  occaFinish();

  for (i = 0; i < entries; ++i) {
    printf("%d = %f\n", i, ab[i]);
  }
  for (i = 0; i < entries; ++i) {
    if (fabs(ab[i] - (a[i] + b[i])) > 1.0e-8) {
      exit(1);
    }
  }

  occaFree(&args);
  occaFree(&addVectors);
  occaFreeUvaPtr(a);
  occaFreeUvaPtr(b);
  occaFreeUvaPtr(ab);

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
    "      default_value: { mode: 'Serial' },"
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
