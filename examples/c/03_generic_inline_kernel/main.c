#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <occa.h>
#include <occa/c/experimental.h>

//---[ Internal Tools ]-----------------
// Note: These headers are not officially supported
//       Please don't rely on it outside of the occa examples
#include <occa/internal/c/cli.h>
//======================================

occaJson parseArgs(int argc, const char **argv);

int main(int argc, const char **argv) {
  occaJson args = parseArgs(argc, argv);

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

  occaJson props = occaCreateJson();
  occaJsonObjectSet(props,
                    "defines/TILE_SIZE",
                    occaInt(16));

  occaScope scope = occaCreateScope(props);

  // Build the variable scope used inside the inlined OKL code
  occaScopeAddConst(scope, "entries", occaInt(entries));
  occaScopeAddConst(scope, "a", o_a);
  occaScopeAddConst(scope, "b", o_b);
  // We can name our scoped variales anything
  occaScopeAdd(scope, "output", o_ab);
  // We can also add unused variables to the scope which could be
  // useful while debugging
  occaScopeAdd(scope, "debugValue", occaInt(42));

  OCCA_JIT(
    scope,
    (
      // TILE_SIZE is passed as a compile-time define as opposed to a runtime variable
      // through props
      for (int i = 0; i < entries; ++i; @tile(TILE_SIZE, @outer, @inner)) {
        // Note it's using the scope name 'output' and not its original value name 'ab'
        output[i] = a[i] + b[i];
      }
    )
  );

  // Copy result to the host
  occaCopyMemToPtr(ab, o_ab, occaAllBytes, 0, occaDefault);

  for (i = 0; i < entries; ++i) {
    printf("%d = %f\n", i, ab[i]);
  }
  for (i = 0; i < entries; ++i) {
    if (fabs(ab[i] - (a[i] + b[i])) > 1.0e-8) {
      exit(1);
    }
  }

  // Free host memory
  free(a);
  free(b);
  free(ab);

  // Free device memory and occa objects
  occaFree(&args);
  occaFree(&props);
  occaFree(&scope);
  occaFree(&o_a);
  occaFree(&o_b);
  occaFree(&o_ab);

  return 0;
}

occaJson parseArgs(int argc, const char **argv) {
  occaJson args = occaCliParseArgs(
    argc, argv,
    "{"
    "  description: 'Example showing inline OKL code',"
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
