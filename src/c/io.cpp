#include <stdarg.h>

#include <occa/c/types.hpp>
#include <occa/c/io.h>
#include <occa/io/output.hpp>

OCCA_START_EXTERN_C

void OCCA_RFUNC occaOverrideStdout(occaIoOutputFunction_t out) {
  occa::io::stdout.setOverride(out);
}

void OCCA_RFUNC occaOverrideStderr(occaIoOutputFunction_t out) {
  occa::io::stderr.setOverride(out);
}

OCCA_END_EXTERN_C
