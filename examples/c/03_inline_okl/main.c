#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <occa.h>

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

  float *a  = (float*) occaTypedUMalloc(entries, occaDtypeFloat, NULL, occaDefault);
  float *b  = (float*) occaTypedUMalloc(entries, occaDtypeFloat, NULL, occaDefault);
  float *ab = (float*) occaTypedUMalloc(entries, occaDtypeFloat, NULL, occaDefault);

  for (i = 0; i < entries; ++i) {
    a[i]  = i;
    b[i]  = 1 - i;
    ab[i] = 0;
  }

  occaProperties props = occaCreateProperties();
  occaPropertiesSet(props,
                    "defines/TILE_SIZE",
                    occaInt(16));

  occaScope scope = occaCreateScope(props);

  // Build the variable scope used inside the inlined OKL code
  occaScopeAddConst(scope, "entries", occaInt(entries));
  occaScopeAddConst(scope, "a", occaPtr(a));
  occaScopeAddConst(scope, "b", occaPtr(b));
  // We can name our scoped variales anything
  occaScopeAdd(scope, "output", occaPtr(ab));
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
  occaFree(&props);
  occaFree(&scope);
  occaFreeUvaPtr(a);
  occaFreeUvaPtr(b);
  occaFreeUvaPtr(ab);

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
