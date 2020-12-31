#include <stdarg.h>

#include <occa/internal/c/types.hpp>
#include <occa/c/io.h>
#include <occa/internal/io/output.hpp>

OCCA_START_EXTERN_C

void occaOverrideStdout(occaIoOutputFunction_t out) {
  occa::io::stdout.setOverride(out);
}

void occaOverrideStderr(occaIoOutputFunction_t out) {
  occa::io::stderr.setOverride(out);
}

OCCA_END_EXTERN_C
