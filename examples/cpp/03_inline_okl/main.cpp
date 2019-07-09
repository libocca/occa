#include <iostream>

#include <occa.hpp>

occa::json parseArgs(int argc, const char **argv);

int main(int argc, const char **argv) {
  occa::json args = parseArgs(argc, argv);

  occa::setDevice((std::string) args["options/device"]);

  int entries = 5;

  float *a  = occa::umalloc<float>(entries);
  float *b  = occa::umalloc<float>(entries);
  float *ab = occa::umalloc<float>(entries);

  for (int i = 0; i < entries; ++i) {
    a[i]  = i;
    b[i]  = 1 - i;
    ab[i] = 0;
  }

  occa::properties props;
  props["defines/TILE_SIZE"] = 16;

  // Const-ness of variables is passed through which can be useful for the compiler
  const float *c_a = a;
  const float *c_b = b;

  // Arguments:
  // 1. Runtime occa::properties (pass occa::properties() to ignore this argument)
  // 2. Captured variables
  // 3. Inlined OKL source
  INLINE_OKL(
    props,
    (entries, c_a, c_b, ab),
    (
      // TILE_SIZE is passed as a compile-time define as opposed to a runtime variable
      // through props
      for (int i = 0; i < entries; ++i; @tile(TILE_SIZE, @outer, @inner)) {
        ab[i] = 100 * (c_a[i] + c_b[i]);
      }
    )
  );

  // Alternatively, occa::scope can be used to more finely capture variables
  occa::scope scope(props);

  // Build the variable scope used inside the inlined OKL code
  scope.addConst("entries", entries);
  scope.addConst("a", a);
  scope.addConst("b", b);

  // We can name our scoped variales anything
  scope.add("output", ab);

  // We can also add unused variables to the scope which could be
  // useful while debugging
  scope.add("debugValue", 42);

  INLINE_OKL_WITH_SCOPE(
    scope,
    (
      for (int i = 0; i < entries; ++i; @tile(TILE_SIZE, @outer, @inner)) {
        // Note it's using the scope name 'output' and not its original value name 'ab'
        output[i] = a[i] + b[i];
      }
    )
  );

  occa::finish();

  for (int i = 0; i < entries; ++i) {
    std::cout << i << ": " << ab[i] << '\n';
  }
  for (int i = 0; i < entries; ++i) {
    if (ab[i] != (a[i] + b[i]))
      throw 1;
  }

  occa::free(a);
  occa::free(b);
  occa::free(ab);

  return 0;
}

occa::json parseArgs(int argc, const char **argv) {
  // Note:
  //   occa::cli is not supported yet, please don't rely on it
  //   outside of the occa examples
  occa::cli::parser parser;
  parser
    .withDescription(
      "Example showing inline OKL code"
    )
    .addOption(
      occa::cli::option('d', "device",
                        "Device properties (default: \"mode: 'Serial'\")")
      .withArg()
      .withDefaultValue("mode: 'Serial'")
    )
    .addOption(
      occa::cli::option('v', "verbose",
                        "Compile kernels in verbose mode")
    );

  occa::json args = parser.parseArgs(argc, argv);
  occa::settings()["kernel/verbose"] = args["options/verbose"];

  return args;
}
