#include "occa/parser/parser.hpp"

using namespace occa;
using namespace occa::parserNS;

int main(int argc, char **argv){
  occa::parser parser;
  std::string parsedContent;

  parser.warnForMissingBarriers     = false;
  parser.warnForBarrierConditionals = false;

  strToStrMap_t compilerFlags;
  // compilerFlags["mode"] = "Serial";
  compilerFlags["mode"] = "OpenCL";

  compilerFlags["language"] = "C";
  // parsedContent = parser.parseFile("tests/easy.c"             , compilerFlags);
  // parsedContent = parser.parseFile("tests/test.cpp"           , compilerFlags);
  // parsedContent = parser.parseFile("tests/test2.cpp"          , compilerFlags);
  parsedContent = parser.parseFile("tests/scratch.okl"        , compilerFlags);
  // parsedContent = parser.parseFile("tests/pwdg.okl"           , compilerFlags);
  // parsedContent = parser.parseFile("tests/openclTest.cpp"     , compilerFlags);
  // parsedContent = parser.parseFile("tests/cudaTest.cpp"       , compilerFlags);
  // parsedContent = parser.parseFile("tests/fd2d_cuda.okl"      , compilerFlags);
  // parsedContent = parser.parseFile("tests/midg.okl"           , compilerFlags);
  // parsedContent = parser.parseFile("tests/cleanTest.c"        , compilerFlags);
  // parsedContent = parser.parseFile("tests/clangTest2.c"       , compilerFlags);
  // parsedContent = parser.parseFile("tests/addVectors.okl"     , compilerFlags);
  // parsedContent = parser.parseFile("tests/PCGpart1.cl"        , compilerFlags);
  // parsedContent = parser.parseFile("tests/lookup_kernel.okl"  , compilerFlags);
  // parsedContent = parser.parseFile("tests/reduction.cl"       , compilerFlags);
  // parsedContent = parser.parseFile("tests/loopy.cl"           , compilerFlags);
  // parsedContent = parser.parseFile("tests/addVectors_loopy.cl", compilerFlags);

  //---[ Fortran ]----------------------
  // compilerFlags["language"] = "Fortran";

  // parsedContent = parser.parseFile("tests/scratch.ofl"   , compilerFlags);
  // parsedContent = parser.parseFile("tests/addVectors.ofl", compilerFlags);
  // parsedContent = parser.parseFile("tests/fakeNuma.ofl"  , compilerFlags);
  //====================================

  //---[ Magic ]------------------------
  // compilerFlags["magic"] = "enabled";

  //---[ Generic ]--------------------
  // parsedContent = parser.parseFile("tests/ridgV.oak", compilerFlags);
  // parsedContent = parser.parseFile("magicTests/addVectors.oak", compilerFlags);
  // parsedContent = parser.parseFile("magicTests/fdTest.oak", compilerFlags);

  //---[ Rodinia ]--------------------
  // parsedContent = parser.parseFile("magicTests/rodinia/backprop.oak", compilerFlags);
  // parsedContent = parser.parseFile("magicTests/rodinia/bfs.oak", compilerFlags); // Fails: Has dynamic bounds
  // parsedContent = parser.parseFile("magicTests/rodinia/gaussian.oak", compilerFlags);

  //---[ Arturo ]---------------------
  // parsedContent = parser.parseFile("magicTests/arturo/hermiteAdvec.okl", compilerFlags);

  //---[ Frank ]----------------------
  // parsedContent = parser.parseFile("magicTests/frank/acousticsSurfaceKernelHex.okl", compilerFlags);
  // parsedContent = parser.parseFile("magicTests/frank/acousticsSurfaceKernelPri.okl", compilerFlags);
  // parsedContent = parser.parseFile("magicTests/frank/acousticsSurfaceKernelPyr.okl", compilerFlags);
  // parsedContent = parser.parseFile("magicTests/frank/acousticsSurfaceKernelTet.okl", compilerFlags);

  // parsedContent = parser.parseFile("magicTests/frank/acousticsUpdateKernelHex.okl", compilerFlags);
  // parsedContent = parser.parseFile("magicTests/frank/acousticsUpdateKernelPri.okl", compilerFlags);
  // parsedContent = parser.parseFile("magicTests/frank/acousticsUpdateKernelPyr.okl", compilerFlags);
  // parsedContent = parser.parseFile("magicTests/frank/acousticsUpdateKernelTet.okl", compilerFlags);

  // parsedContent = parser.parseFile("magicTests/frank/acousticsVolumeKernelHex.okl", compilerFlags);
  // parsedContent = parser.parseFile("magicTests/frank/acousticsVolumeKernelPri.okl", compilerFlags);
  // parsedContent = parser.parseFile("magicTests/frank/acousticsVolumeKernelPyr.okl", compilerFlags);
  // parsedContent = parser.parseFile("magicTests/frank/acousticsVolumeKernelTet.okl", compilerFlags);

  // parsedContent = parser.parseFile("magicTests/frank/genericPartialGetKernel.okl", compilerFlags);
  // parsedContent = parser.parseFile("magicTests/frank/genericPartialPutKernel.okl", compilerFlags);

  //---[ Reid ]-----------------------
  // parsedContent = parser.parseFile("magicTests/reid/scvadd.okl", compilerFlags);
  // parsedContent = parser.parseFile("magicTests/reid/scvdiv.okl", compilerFlags);
  // parsedContent = parser.parseFile("magicTests/reid/scvmult.okl", compilerFlags);
  // parsedContent = parser.parseFile("magicTests/reid/test_complex_ips.okl", compilerFlags);
  // parsedContent = parser.parseFile("magicTests/reid/test_ips.okl", compilerFlags);
  // parsedContent = parser.parseFile("magicTests/reid/test_complex_pwdg.okl", compilerFlags); // Fails: Doesn't handle structs yet
  // parsedContent = parser.parseFile("magicTests/reid/test_pwdg.okl", compilerFlags);         // Fails: Doesn't handle structs yet

  //---[ Jesse ]----------------------
  // parsedContent = parser.parseFile("magicTests/jesse/compute_error.okl", compilerFlags);
  // parsedContent = parser.parseFile("magicTests/jesse/compute_u_surface.okl", compilerFlags);
  // parsedContent = parser.parseFile("magicTests/jesse/invertMass.okl", compilerFlags);
  // parsedContent = parser.parseFile("magicTests/jesse/rk_step_kernel.okl", compilerFlags);
  // parsedContent = parser.parseFile("magicTests/jesse/surface_kernel.okl", compilerFlags);
  // parsedContent = parser.parseFile("magicTests/jesse/volume_kernel.okl", compilerFlags);

  //---[ Axel ]-----------------------
  // parsedContent = parser.parseFile("magicTests/axel/acousticsPmlSourceViaInterfKernel.okl", compilerFlags); // Fails: Simplification expansion get's too big and loops?
  // parsedContent = parser.parseFile("magicTests/axel/acousticsForcingSurfaceKernel.okl", compilerFlags);     // Fails: face[f] in array access
  // parsedContent = parser.parseFile("magicTests/axel/acousticsPmlSurfaceKernel.okl", compilerFlags);         // Fails: Switch
  // parsedContent = parser.parseFile("magicTests/axel/acousticsSourceViaInterfKernel.okl", compilerFlags);    // Fails: Simplification expansion get's too big and loops?
  // parsedContent = parser.parseFile("magicTests/axel/acousticsPmlForcingSurfaceKernel.okl", compilerFlags);  // Fails: face[f] in array access
  // parsedContent = parser.parseFile("magicTests/axel/acousticsSurfaceKernel.okl", compilerFlags);            // Fails: Switch

  // parsedContent = parser.parseFile("magicTests/axel/acousticsCorrelationKernel.okl", compilerFlags);
  // parsedContent = parser.parseFile("magicTests/axel/acousticsPmlVolumeKernel.okl", compilerFlags);
  // parsedContent = parser.parseFile("magicTests/axel/acousticsUpdateKernel.okl", compilerFlags);
  // parsedContent = parser.parseFile("magicTests/axel/genericPartialGetKernel.okl", compilerFlags);
  // parsedContent = parser.parseFile("magicTests/axel/acousticsVolumeKernel.okl", compilerFlags);
  // parsedContent = parser.parseFile("magicTests/axel/acousticsPmlUpdateKernel.okl", compilerFlags);
  // parsedContent = parser.parseFile("magicTests/axel/genericPartialPutKernel.okl", compilerFlags);
  // parsedContent = parser.parseFile("magicTests/axel/genericMaskKernel.okl", compilerFlags);
  // parsedContent = parser.parseFile("magicTests/axel/genericSourceKernel.okl", compilerFlags);
  //====================================

  std::cout << parsedContent << '\n';
}
