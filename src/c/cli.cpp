#include <occa/c/cli.h>
#include <occa/c/types.hpp>
#include <occa/tools/cli.hpp>

OCCA_START_EXTERN_C

occaJson occaCliParseArgs(const int argc,
                          const char **argv,
                          const char *config) {
  return occa::c::newOccaType(
    *(new occa::json(occa::cli::parse(argc, argv, config))),
    true
  );
}

OCCA_END_EXTERN_C
