#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

#include <stdlib.h>

inline std::string readFile(const std::string &filename){
  std::ifstream fs(filename.c_str());
  return std::string(std::istreambuf_iterator<char>(fs),
                     std::istreambuf_iterator<char>());
}

inline std::string saveFileToVariable(std::string filename,
                                      std::string varName,
                                      int &chars,
                                      std::string indent = ""){
  std::stringstream occaDeviceDefines;
  occaDeviceDefines << "#define OCCA_USING_CPU 0\n"
                    << "#define OCCA_USING_GPU 0\n"
                    << '\n'
                    << "#define OCCA_USING_PTHREADS 0\n"
                    << "#define OCCA_USING_OPENMP   0\n"
                    << "#define OCCA_USING_OPENCL   0\n"
                    << "#define OCCA_USING_CUDA     0\n"
                    << "#define OCCA_USING_COI      0\n";

    std::string fileContents = occaDeviceDefines.str() + readFile(filename);
  chars = fileContents.size();

  std::stringstream headerSS;

  headerSS << indent << "char " << varName << "[" << chars + 1 << "] = {";
  std::string header = headerSS.str();

  std::string tab;
  tab.assign(header.size(), ' ');

  std::stringstream ss;

  ss << header;

  ss << std::showbase
     << std::internal
     << std::setfill('0');

  for(int i = 0; i < chars; ++i){
    ss << std::hex << std::setw(4) << (int) fileContents[i] << ", ";

    if((i % 8) == 7)
      ss << '\n' << indent << tab;
  }

  ss << std::hex << std::setw(4) << 0 << "};\n";

  return ss.str();
}

int main(int argc, char **argv){
  int mpChars, clChars, cuChars, ptChars, coiChars, coiMainChars;

  if(argc != 2){
    std::cout << "Usage " << argv[0] << " occa_dir"  ;
    throw 1;
  }
  std::string occaDir(argv[1]);

  std::string ns = "namespace occa {";


  std::string pt = saveFileToVariable(occaDir + "/include/defines/occaPthreadsDefines.hpp",
                                      "occaPthreadsDefines",
                                      ptChars,
                                      "    ");

  std::string mp = saveFileToVariable(occaDir + "/include/defines/occaOpenMPDefines.hpp",
                                      "occaOpenMPDefines",
                                      mpChars,
                                      "    ");

  std::string cl = saveFileToVariable(occaDir + "/include/defines/occaOpenCLDefines.hpp",
                                      "occaOpenCLDefines",
                                      clChars,
                                      "    ");

  std::string cu = saveFileToVariable(occaDir + "/include/defines/occaCUDADefines.hpp",
                                      "occaCUDADefines",
                                      cuChars,
                                      "    ");

  std::string coi = saveFileToVariable(occaDir + "/include/defines/occaCOIDefines.hpp",
                                       "occaCOIDefines",
                                       coiChars,
                                       "    ");

  std::string coiMain = saveFileToVariable(occaDir + "/include/defines/occaCOIMain.hpp",
                                           "occaCOIMain",
                                           coiMainChars,
                                           "    ");

  std::string occaKernelDefinesHeader = occaDir + "/include/occaKernelDefines.hpp";
  std::string occaKernelDefinesSource = occaDir + "/src/occaKernelDefines.cpp";

  std::ofstream fs;
  fs.open(occaKernelDefinesHeader.c_str());

  fs << ns << '\n'
     << "    extern char occaPthreadsDefines[" << ptChars << "];\n"
     << "    extern char occaOpenMPDefines[" << mpChars  << "];\n"
     << "    extern char occaOpenCLDefines[" << clChars  << "];\n"
     << "    extern char occaCUDADefines["   << cuChars  << "];\n"
     << "    extern char occaCOIDefines["   << coiChars << "];\n"
     << "    extern char occaCOIMain["   << coiMainChars << "];\n"
     << "}\n";

  fs.close();

  fs.open(occaKernelDefinesSource.c_str());

  fs << ns << '\n'
     << pt << '\n'
     << mp << '\n'
     << cl << '\n'
     << cu << '\n'
     << coi << '\n'
     << coiMain << '\n'
     << "}\n";

  fs.close();
}
