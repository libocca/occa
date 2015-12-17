#include "occa/parser/parser.hpp"

using namespace occa;
using namespace occa::parserNS;

int main(int argc, char **argv){
  occa::parser parser;
  std::string parsedContent;

  occa::flags_t parserFlags;

  parserFlags["mode"] = "Serial";
  // parserFlags["mode"] = "OpenCL";

  parserFlags["warn-for-missing-barriers"]     = "no";
  parserFlags["warn-for-conditional-barriers"] = "no";
  parserFlags["automate-add-barriers"]         = "yes";

  parserFlags["language"] = "C";
  // parsedContent = parser.parseFile("tests/easy.c"             , parserFlags);
  // parsedContent = parser.parseFile("tests/test.cpp"           , parserFlags);
  // parsedContent = parser.parseFile("tests/test2.cpp"          , parserFlags);
  parsedContent = parser.parseFile("tests/scratch.okl"        , parserFlags);
  // parsedContent = parser.parseFile("tests/pwdg.okl"           , parserFlags);
  // parsedContent = parser.parseFile("tests/openclTest.cpp"     , parserFlags);
  // parsedContent = parser.parseFile("tests/cudaTest.cpp"       , parserFlags);
  // parsedContent = parser.parseFile("tests/fd2d_cuda.okl"      , parserFlags);
  // parsedContent = parser.parseFile("tests/midg.okl"           , parserFlags);
  // parsedContent = parser.parseFile("tests/cleanTest.c"        , parserFlags);
  // parsedContent = parser.parseFile("tests/clangTest2.c"       , parserFlags);
  // parsedContent = parser.parseFile("tests/addVectors.okl"     , parserFlags);
  // parsedContent = parser.parseFile("tests/PCGpart1.cl"        , parserFlags);
  // parsedContent = parser.parseFile("tests/lookup_kernel.okl"  , parserFlags);
  // parsedContent = parser.parseFile("tests/reduction.cl"       , parserFlags);
  // parsedContent = parser.parseFile("tests/loopy.cl"           , parserFlags);
  // parsedContent = parser.parseFile("tests/addVectors_loopy.cl", parserFlags);

  //---[ Fortran ]----------------------
  // parserFlags["language"] = "Fortran";

  // parsedContent = parser.parseFile("tests/scratch.ofl"   , parserFlags);
  // parsedContent = parser.parseFile("tests/addVectors.ofl", parserFlags);
  // parsedContent = parser.parseFile("tests/fakeNuma.ofl"  , parserFlags);
  //====================================

  //---[ Magic ]------------------------
  // parserFlags["magic"] = "enabled";

  //---[ Generic ]--------------------
  // parsedContent = parser.parseFile("tests/ridgV.oak", parserFlags);
  // parsedContent = parser.parseFile("magicTests/addVectors.oak", parserFlags);
  // parsedContent = parser.parseFile("magicTests/fdTest.oak", parserFlags);

  //---[ Rodinia ]--------------------
  // parsedContent = parser.parseFile("magicTests/rodinia/backprop.oak", parserFlags);
  // parsedContent = parser.parseFile("magicTests/rodinia/bfs.oak", parserFlags); // Fails: Has dynamic bounds
  // parsedContent = parser.parseFile("magicTests/rodinia/gaussian.oak", parserFlags);

  //---[ Arturo ]---------------------
  // parsedContent = parser.parseFile("magicTests/arturo/hermiteAdvec.okl", parserFlags);

  //---[ Frank ]----------------------
  // parsedContent = parser.parseFile("magicTests/frank/acousticsSurfaceKernelHex.okl", parserFlags);
  // parsedContent = parser.parseFile("magicTests/frank/acousticsSurfaceKernelPri.okl", parserFlags);
  // parsedContent = parser.parseFile("magicTests/frank/acousticsSurfaceKernelPyr.okl", parserFlags);
  // parsedContent = parser.parseFile("magicTests/frank/acousticsSurfaceKernelTet.okl", parserFlags);

  // parsedContent = parser.parseFile("magicTests/frank/acousticsUpdateKernelHex.okl", parserFlags);
  // parsedContent = parser.parseFile("magicTests/frank/acousticsUpdateKernelPri.okl", parserFlags);
  // parsedContent = parser.parseFile("magicTests/frank/acousticsUpdateKernelPyr.okl", parserFlags);
  // parsedContent = parser.parseFile("magicTests/frank/acousticsUpdateKernelTet.okl", parserFlags);

  // parsedContent = parser.parseFile("magicTests/frank/acousticsVolumeKernelHex.okl", parserFlags);
  // parsedContent = parser.parseFile("magicTests/frank/acousticsVolumeKernelPri.okl", parserFlags);
  // parsedContent = parser.parseFile("magicTests/frank/acousticsVolumeKernelPyr.okl", parserFlags);
  // parsedContent = parser.parseFile("magicTests/frank/acousticsVolumeKernelTet.okl", parserFlags);

  // parsedContent = parser.parseFile("magicTests/frank/genericPartialGetKernel.okl", parserFlags);
  // parsedContent = parser.parseFile("magicTests/frank/genericPartialPutKernel.okl", parserFlags);

  //---[ Reid ]-----------------------
  // parsedContent = parser.parseFile("magicTests/reid/scvadd.okl", parserFlags);
  // parsedContent = parser.parseFile("magicTests/reid/scvdiv.okl", parserFlags);
  // parsedContent = parser.parseFile("magicTests/reid/scvmult.okl", parserFlags);
  // parsedContent = parser.parseFile("magicTests/reid/test_complex_ips.okl", parserFlags);
  // parsedContent = parser.parseFile("magicTests/reid/test_ips.okl", parserFlags);
  // parsedContent = parser.parseFile("magicTests/reid/test_complex_pwdg.okl", parserFlags); // Fails: Doesn't handle structs yet
  // parsedContent = parser.parseFile("magicTests/reid/test_pwdg.okl", parserFlags);         // Fails: Doesn't handle structs yet

  //---[ Jesse ]----------------------
  // parsedContent = parser.parseFile("magicTests/jesse/compute_error.okl", parserFlags);
  // parsedContent = parser.parseFile("magicTests/jesse/compute_u_surface.okl", parserFlags);
  // parsedContent = parser.parseFile("magicTests/jesse/invertMass.okl", parserFlags);
  // parsedContent = parser.parseFile("magicTests/jesse/rk_step_kernel.okl", parserFlags);
  // parsedContent = parser.parseFile("magicTests/jesse/surface_kernel.okl", parserFlags);
  // parsedContent = parser.parseFile("magicTests/jesse/volume_kernel.okl", parserFlags);

  //---[ Axel ]-----------------------
  // parsedContent = parser.parseFile("magicTests/axel/acousticsPmlSourceViaInterfKernel.okl", parserFlags); // Fails: Simplification expansion get's too big and loops?
  // parsedContent = parser.parseFile("magicTests/axel/acousticsForcingSurfaceKernel.okl", parserFlags);     // Fails: face[f] in array access
  // parsedContent = parser.parseFile("magicTests/axel/acousticsPmlSurfaceKernel.okl", parserFlags);         // Fails: Switch
  // parsedContent = parser.parseFile("magicTests/axel/acousticsSourceViaInterfKernel.okl", parserFlags);    // Fails: Simplification expansion get's too big and loops?
  // parsedContent = parser.parseFile("magicTests/axel/acousticsPmlForcingSurfaceKernel.okl", parserFlags);  // Fails: face[f] in array access
  // parsedContent = parser.parseFile("magicTests/axel/acousticsSurfaceKernel.okl", parserFlags);            // Fails: Switch

  // parsedContent = parser.parseFile("magicTests/axel/acousticsCorrelationKernel.okl", parserFlags);
  // parsedContent = parser.parseFile("magicTests/axel/acousticsPmlVolumeKernel.okl", parserFlags);
  // parsedContent = parser.parseFile("magicTests/axel/acousticsUpdateKernel.okl", parserFlags);
  // parsedContent = parser.parseFile("magicTests/axel/genericPartialGetKernel.okl", parserFlags);
  // parsedContent = parser.parseFile("magicTests/axel/acousticsVolumeKernel.okl", parserFlags);
  // parsedContent = parser.parseFile("magicTests/axel/acousticsPmlUpdateKernel.okl", parserFlags);
  // parsedContent = parser.parseFile("magicTests/axel/genericPartialPutKernel.okl", parserFlags);
  // parsedContent = parser.parseFile("magicTests/axel/genericMaskKernel.okl", parserFlags);
  // parsedContent = parser.parseFile("magicTests/axel/genericSourceKernel.okl", parserFlags);
  //====================================

  std::cout << parsedContent << '\n';
}
