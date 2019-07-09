#include <occa/c/types.hpp>
#include <occa/c/kernelBuilder.h>
#include <occa/core/kernelBuilder.hpp>

OCCA_START_EXTERN_C

occaKernelBuilder OCCA_RFUNC occaKernelBuilderFromInlinedOkl(
  occaScope scope,
  const char *kernelSource,
  const char *kernelName
) {
  occa::kernelBuilder kb = occa::kernelBuilder::fromString(
    occa::formatInlinedKernelFromScope(occa::c::scope(scope),
                                       kernelSource,
                                       kernelName),
    kernelName
  );

  return occa::c::newOccaType(*(new occa::kernelBuilder(kb)));
}

void OCCA_RFUNC occaKernelBuilderRun(
  occaKernelBuilder kernelBuilder,
  occaScope scope
) {
  occa::c::kernelBuilder(kernelBuilder).run(
    occa::c::scope(scope)
  );
}

OCCA_END_EXTERN_C
