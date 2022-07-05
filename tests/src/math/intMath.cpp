#include <iostream>
#include <vector>

#include "occa.hpp"

std::vector<std::string> arg_types = {"int"};

std::string unary_args = "x";
std::string binary_args = "x,y";

std::vector<std::string> unary_functions = {"abs"};

std::vector<std::string> binary_functions = {"max","min"};

std::string kernel_front_half =
"@kernel \n"
"void f(const int dummy_arg) { \n"
"  @outer \n"
"  for (int b=0; b<1; ++b) { \n"
"    @inner \n"
"    for (int t=0; t<1; ++t) { \n"
;

std::string kernel_back_half =
"    } \n"
"  } \n"
"} \n"
;

void testUnaryFunctions(const occa::device& d) {
  for (auto&& int_type : arg_types) {
    const std::string arg_decl = 
      "        " + int_type + " " + unary_args + "; \n";
    for (auto&& func : unary_functions) {
      const std::string function_call = 
        "        " + int_type + " w = " + func + "(" + unary_args + "); \n";
      const std::string kernel_src = 
        kernel_front_half + arg_decl + function_call +kernel_back_half;

      occa::kernel k = d.buildKernelFromString(kernel_src,"f");
    }
  }
}

void testBinaryFunctions(const occa::device& d) {
  for (auto&& int_type : arg_types) {
    const std::string arg_decl = 
      "        " + int_type + " " + binary_args + "; \n";
    for (auto&& func : binary_functions) {
      const std::string function_call = 
        "        " + int_type + " w = " + func + "(" + binary_args + "); \n";
      const std::string kernel_src = 
        kernel_front_half + arg_decl + function_call +kernel_back_half;

      occa::kernel k = d.buildKernelFromString(kernel_src,"f");
    }
  }
}

int main() {
  std::vector<occa::device> devices = {
    occa::device({{"mode", "Serial"}}),
    occa::device({{"mode", "OpenMP"}}),
    occa::device({{"mode", "CUDA"},{"device_id", 0}}),
    occa::device({{"mode", "HIP"},{"device_id", 0}}),
    occa::device({{"mode", "OpenCL"},{"platform_id",0},{"device_id", 0}}),
    occa::device({{"mode", "dpcpp"},{"platform_id",0},{"device_id", 0}})
  };

  for(auto &d : devices) {
    std::cout << "Testing mode: " << d.mode() << "\n";
    testUnaryFunctions(d);
    testBinaryFunctions(d);
  }

  return 0;
}
