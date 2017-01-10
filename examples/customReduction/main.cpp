#include <iostream>

#include "occa.hpp"

class customReduction {
public:
  bool init;
  occa::kernel kernel;

  customReduction() {
    init = false;
  }

  customReduction(occa::device device,
                  const std::string &inType,
                  const std::string &inName,
                  const std::string &outType,
                  const std::string &outName,
                  const std::string &equation) {
    init = false;
    setup(device,
          inType, inName,
          outType, outName,
          equation);
  }

  void setup(occa::device device,
             const std::string &inType,
             const std::string &inName,
             const std::string &outType,
             const std::string &outName,
             const std::string &equation) {

    std::stringstream ss;

    ss << "kernel void customReductionKernel(const int entries,\n"
       << "                                  const " << inType  << " *custom_reduction_in ,\n"
       << "                                        " << outType << " *custom_reduction_out) {\n"
       << "  for (int group = 0; group < entries; group += 128; outer0) {\n"
       << "    for (int item = group; item < (group + 128); ++item; inner0) {\n"
       << "      const int n = item;\n\n"

       << "      if (n < entries) {\n"
       << "        const " << inType  << ' ' << inName  << " = custom_reduction_in[item];\n"
       << "              " << outType << ' ' << outName << ";\n"
       << "        " << equation << '\n'
       << "        custom_reduction_out[item] = " << outName << ";\n"
       << "      }\n"
       << "    }\n"
       << "  }\n"
       << '}';

    if (init) {
      kernel.free();
    }
    kernel = device.buildKernelFromString(ss.str(),
                                          "customReductionKernel",
                                          occa::properties("language = OKL"));

    init = true;
  }

  void operator () (const int entries, occa::memory &in, occa::memory &out) {
    kernel(entries, in, out);
  }

  void free() {
    if (init) {
      kernel.free();
    }
    init = false;
  }
};

int main(int argc, char **argv) {
  int entries = 5;

  float *a  = new float[entries];
  float *a2 = new float[entries];

  for (int i = 0; i < entries; ++i) {
    a[i]  = i;
    a2[i] = 0;
  }

  occa::device device("mode: 'Serial'");
  customReduction squareArray;
  occa::memory o_a, o_a2;

  o_a  = device.malloc(entries*sizeof(float));
  o_a2 = device.malloc(entries*sizeof(float));

  squareArray.setup(device,
                    "float", "a",
                    "float", "a2",
                    "a2 = a * a;");

  o_a.copyFrom(a);
  o_a2.copyFrom(a2);

  squareArray(entries, o_a, o_a2);

  o_a2.copyTo(a2);

  for (int i = 0; i < 5; ++i)
    std::cout << a[i] << "^2 = " << a2[i] << '\n';

  for (int i = 0; i < entries; ++i) {
    if (a2[i] != (a[i] * a[i]))
      throw 1;
  }

  delete [] a;
  delete [] a2;

  squareArray.free();
  o_a.free();
  o_a2.free();
  device.free();

  return 0;
}
