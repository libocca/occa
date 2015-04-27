#include "occaParser.hpp"

using namespace occa;
using namespace occa::parserNS;

int main(int argc, char **argv){
  occa::parser parser;
  parser.warnForMissingBarriers     = false;
  parser.warnForBarrierConditionals = false;

  // {
  //   std::string parsedContent = parser.parseFile("tests/easy.c");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   std::string parsedContent = parser.parseFile("tests/test.cpp");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   std::string parsedContent = parser.parseFile("tests/scratch.okl");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   std::string parsedContent = parser.parseFile("tests/pwdg.okl");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   std::string parsedContent = parser.parseFile("tests/openclTest.cpp");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   std::string parsedContent = parser.parseFile("tests/cudaTest.cpp");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   std::string parsedContent = parser.parseFile("tests/fd2d_cuda.okl");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   std::string parsedContent = parser.parseFile("tests/midg.okl");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   std::string parsedContent = parser.parseFile("tests/cleanTest.c");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   std::string parsedContent = parser.parseFile("tests/clangTest.c");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   std::string parsedContent = parser.parseFile("tests/addVectors.okl");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   std::string parsedContent = parser.parseFile("tests/PCGpart1.cl");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   std::string parsedContent = parser.parseFile("tests/lookup_kernel.okl");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   std::string parsedContent = parser.parseFile("tests/reduction.cl");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   std::string parsedContent = parser.parseFile("tests/loopy.cl");
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   std::string parsedContent = parser.parseFile("tests/addVectors_loopy.cl");
  //   std::cout << parsedContent << '\n';
  // }

  //---[ Fortran ]----------------------
  // {
  //   std::string parsedContent = parser.parseFile("tests/scratch.ofl",
  //                                                occa::parserNS::parsingFortran);
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   std::string parsedContent = parser.parseFile("tests/addVectors.ofl",
  //                                                occa::parserNS::parsingFortran);
  //   std::cout << parsedContent << '\n';
  // }

  // {
  //   std::string parsedContent = parser.parseFile("tests/fakeNuma.ofl",
  //                                                occa::parserNS::parsingFortran);
  //   std::cout << parsedContent << '\n';
  // }
  //====================================

  //---[ Magic ]------------------------
  // {
  //   parser.magicEnabled = true;

  //   std::string parsedContent = parser.parseFile("tests/ridgV.oak");
  //   std::cout << parsedContent << '\n';
  // }
  {
    parser.magicEnabled = true;

    //---[ Arturo ]---------------------
    // std::string parsedContent = parser.parseFile("magicTests/arturo/hermiteAdvec.okl");

    //---[ Frank ]----------------------
    // std::string parsedContent = parser.parseFile("magicTests/frank/acousticsSurfaceKernelHex.okl");
    // std::string parsedContent = parser.parseFile("magicTests/frank/acousticsSurfaceKernelPri.okl");
    // std::string parsedContent = parser.parseFile("magicTests/frank/acousticsSurfaceKernelPyr.okl");
    // std::string parsedContent = parser.parseFile("magicTests/frank/acousticsSurfaceKernelTet.okl");

    // std::string parsedContent = parser.parseFile("magicTests/frank/acousticsUpdateKernelHex.okl");
    // std::string parsedContent = parser.parseFile("magicTests/frank/acousticsUpdateKernelPri.okl");
    // std::string parsedContent = parser.parseFile("magicTests/frank/acousticsUpdateKernelPyr.okl");
    // std::string parsedContent = parser.parseFile("magicTests/frank/acousticsUpdateKernelTet.okl");

    // std::string parsedContent = parser.parseFile("magicTests/frank/acousticsVolumeKernelHex.okl");
    // std::string parsedContent = parser.parseFile("magicTests/frank/acousticsVolumeKernelPri.okl");
    // std::string parsedContent = parser.parseFile("magicTests/frank/acousticsVolumeKernelPyr.okl");
    // std::string parsedContent = parser.parseFile("magicTests/frank/acousticsVolumeKernelTet.okl");

    // std::string parsedContent = parser.parseFile("magicTests/frank/genericPartialGetKernel.okl");
    // std::string parsedContent = parser.parseFile("magicTests/frank/genericPartialPutKernel.okl");

    //---[ Reid ]-----------------------
    // std::string parsedContent = parser.parseFile("magicTests/reid/scvadd.okl");
    // std::string parsedContent = parser.parseFile("magicTests/reid/scvdiv.okl");
    // std::string parsedContent = parser.parseFile("magicTests/reid/scvmult.okl");
    // std::string parsedContent = parser.parseFile("magicTests/reid/test_complex_ips.okl");
    // std::string parsedContent = parser.parseFile("magicTests/reid/test_ips.okl");
    // std::string parsedContent = parser.parseFile("magicTests/reid/test_complex_pwdg.okl"); // Fails: Doesn't handle structs yet
    // std::string parsedContent = parser.parseFile("magicTests/reid/test_pwdg.okl");         // Fails: Doesn't handle structs yet

    //---[ Jesse ]----------------------
    // std::string parsedContent = parser.parseFile("magicTests/jesse/compute_error.okl");
    // std::string parsedContent = parser.parseFile("magicTests/jesse/compute_u_surface.okl");
    // std::string parsedContent = parser.parseFile("magicTests/jesse/invertMass.okl");
    // std::string parsedContent = parser.parseFile("magicTests/jesse/rk_step_kernel.okl");
    // std::string parsedContent = parser.parseFile("magicTests/jesse/surface_kernel.okl");
    // std::string parsedContent = parser.parseFile("magicTests/jesse/volume_kernel.okl");

    //---[ Axel ]-----------------------
    // std::string parsedContent = parser.parseFile("magicTests/axel/acousticsPmlSourceViaInterfKernel.okl"); // Fails: Simplification expansion get's too big and loops?
    // std::string parsedContent = parser.parseFile("magicTests/axel/acousticsForcingSurfaceKernel.okl");     // Fails: face[f] in array access
    // std::string parsedContent = parser.parseFile("magicTests/axel/acousticsPmlSurfaceKernel.okl");         // Fails: Switch
    // std::string parsedContent = parser.parseFile("magicTests/axel/acousticsSourceViaInterfKernel.okl");    // Fails: Simplification expansion get's too big and loops?
    // std::string parsedContent = parser.parseFile("magicTests/axel/acousticsPmlForcingSurfaceKernel.okl");  // Fails: face[f] in array access
    // std::string parsedContent = parser.parseFile("magicTests/axel/acousticsSurfaceKernel.okl");            // Fails: Switch

    // std::string parsedContent = parser.parseFile("magicTests/axel/acousticsCorrelationKernel.okl");
    // std::string parsedContent = parser.parseFile("magicTests/axel/acousticsPmlVolumeKernel.okl");
    // std::string parsedContent = parser.parseFile("magicTests/axel/acousticsUpdateKernel.okl");
    // std::string parsedContent = parser.parseFile("magicTests/axel/genericPartialGetKernel.okl");
    // std::string parsedContent = parser.parseFile("magicTests/axel/acousticsVolumeKernel.okl");
    // std::string parsedContent = parser.parseFile("magicTests/axel/acousticsPmlUpdateKernel.okl");
    // std::string parsedContent = parser.parseFile("magicTests/axel/genericPartialPutKernel.okl");
    // std::string parsedContent = parser.parseFile("magicTests/axel/genericMaskKernel.okl");
    // std::string parsedContent = parser.parseFile("magicTests/axel/genericSourceKernel.okl");

    std::cout << parsedContent << '\n';
  }
  //====================================
}
