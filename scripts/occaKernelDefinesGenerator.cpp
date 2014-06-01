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
  std::string fileContents = readFile(filename);
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
  int mpChars, clChars, cuChars;

  char *occaDir_ = getenv("OCCA_DIR");
  if(occaDir_ == NULL){
    std::cout << "Environment variable [OCCA_DIR] is not set.\n";
    throw 1;
  }
  std::string occaDir(occaDir_);

  std::string ns = "namespace occa {";
  std::string mp = saveFileToVariable(occaDir + "/include/occaOpenMPDefines.hpp",
                                      "occaOpenMPDefines",
                                      mpChars,
                                      "    ");

  std::string cl = saveFileToVariable(occaDir + "/include/occaOpenCLDefines.hpp",
                                      "occaOpenCLDefines",
                                      clChars,
                                      "    ");

  std::string cu = saveFileToVariable(occaDir + "/include/occaCUDADefines.hpp",
                                      "occaCUDADefines",
                                      cuChars,
                                      "    ");

  std::string occaKernelDefinesHeader = occaDir + "/include/occaKernelDefines.hpp";
  std::string occaKernelDefinesSource = occaDir + "/src/occaKernelDefines.cpp";

  std::ofstream fs;
  fs.open(occaKernelDefinesHeader.c_str());

  fs << ns << '\n'
     << "    extern char occaOpenMPDefines[" << mpChars << "];\n"
     << "    extern char occaOpenCLDefines[" << clChars << "];\n"
     << "    extern char occaCUDADefines["   << cuChars << "];\n"
     << "}\n";

  fs.close();

  fs.open(occaKernelDefinesSource.c_str());

  fs << ns << '\n'
     << mp << '\n'
     << cl << '\n'
     << cu << '\n'
     << "}\n";

  fs.close();
}
