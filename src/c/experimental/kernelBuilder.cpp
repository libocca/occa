#include <occa/internal/c/types.hpp>
#include <occa/c/experimental/kernelBuilder.h>
#include <occa/experimental/kernelBuilder.hpp>

OCCA_START_EXTERN_C

occaKernelBuilder occaCreateKernelBuilder(
  const char *kernelSource,
  const char *kernelName
) {
  return occa::c::newOccaType(
    *(new occa::kernelBuilder(kernelSource,
                              kernelName))
  );
}

void occaKernelBuilderRun(
  occaKernelBuilder kernelBuilder,
  occaScope scope
) {
  occa::c::kernelBuilder(kernelBuilder).run(
    occa::c::scope(scope)
  );
}

OCCA_END_EXTERN_C
