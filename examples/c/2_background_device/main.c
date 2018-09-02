/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 David Medina and Tim Warburton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */
#include <stdlib.h>
#include <stdio.h>

#include <occa.h>

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

  float *a  = (float*) occaUMalloc(entries * sizeof(float), NULL, occaDefault);
  float *b  = (float*) occaUMalloc(entries * sizeof(float), NULL, occaDefault);
  float *ab = (float*) occaUMalloc(entries * sizeof(float), NULL, occaDefault);

  for (int i = 0; i < entries; ++i) {
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

  for (int i = 0; i < 5; ++i)
    printf("%d = %f\n", i, ab[i]);

  for (int i = 0; i < entries; ++i) {
    if (ab[i] != (a[i] + b[i])) {
      exit(1);
    }
  }

  occaFree(addVectors);
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
    "      description: 'Device properties (default: \"mode: \\'Serial\\'\")',"
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

  occaProperties settings = occaSettings();
  occaPropertiesSet(settings,
                    "kernel/verbose",
                    occaJsonObjectGet(args, "options/verbose", occaBool(0)));

  return args;
}
