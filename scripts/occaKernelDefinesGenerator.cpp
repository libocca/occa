#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

#include <stdlib.h>

inline std::string readFile(const std::string &filename){
  std::ifstream fs(filename.c_str());
  if(!fs) {
	  std::cerr << "unable to open file " << filename;
	  throw 1;
  }
  return std::string(std::istreambuf_iterator<char>(fs),
                     std::istreambuf_iterator<char>());
}

inline std::string saveFileToVariable(std::string filename,
                                      std::string varName,
                                      int &chars,
                                      std::string indent = ""){
  std::stringstream occaDeviceDefines;
  occaDeviceDefines << "#define OCCA_USING_CPU 0" << std::endl
                    << "#define OCCA_USING_GPU 0" << std::endl
					<< std::endl
                    << "#define OCCA_USING_PTHREADS 0" << std::endl
                    << "#define OCCA_USING_OPENMP   0" << std::endl
                    << "#define OCCA_USING_OPENCL   0" << std::endl
                    << "#define OCCA_USING_CUDA     0" << std::endl
                    << "#define OCCA_USING_COI      0" << std::endl;

  std::string occaDefinesFile = readFile(filename);
  std::string fileContents = occaDeviceDefines.str();
  fileContents = fileContents + occaDefinesFile;
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
      ss << std::endl << indent << tab;
  }

  ss << std::hex << std::setw(4) << 0 << "};" << std::endl;

  return ss.str();
}

int main(int argc, char **argv){
  int mpChars, clChars, cuChars, ptChars, coiChars, coiMainChars;

  char *occaDir_ = getenv("OCCA_DIR");
  std::string occaDir;

  if(occaDir_ == NULL)
    occaDir = "..";
  else
    occaDir = occaDir_;

  std::string ns = "namespace occa {";

  std::string mp = saveFileToVariable(occaDir + "/include/occaOpenMPDefines.hpp",
                                      "occaOpenMPDefines",
                                      mpChars,
                                      "    ");

  std::string pt = saveFileToVariable(occaDir + "/include/occaPthreadsDefines.hpp",
                                      "occaPthreadsDefines",
                                      ptChars,
                                      "    ");

  std::string cl = saveFileToVariable(occaDir + "/include/occaOpenCLDefines.hpp",
                                      "occaOpenCLDefines",
                                      clChars,
                                      "    ");

  std::string cu = saveFileToVariable(occaDir + "/include/occaCUDADefines.hpp",
                                      "occaCUDADefines",
                                      cuChars,
                                      "    ");

  std::string coi = saveFileToVariable(occaDir + "/include/occaCOIDefines.hpp",
                                       "occaCOIDefines",
                                       coiChars,
                                       "    ");

  std::string coiMain = saveFileToVariable(occaDir + "/include/occaCOIMain.hpp",
                                           "occaCOIMain",
                                           coiMainChars,
                                           "    ");

  std::string occaKernelDefinesHeader = occaDir + "/include/occaKernelDefines.hpp";
  std::string occaKernelDefinesSource = occaDir + "/src/occaKernelDefines.cpp";

  std::ofstream fs;
  fs.open(occaKernelDefinesHeader.c_str());
  if(!fs) {
	  std::cerr << "unable to open file " << occaKernelDefinesHeader.c_str();
	  throw 1;
  }

  fs << ns << std::endl
	 << "    extern char occaPthreadsDefines[" << ptChars << "];" << std::endl
     << "    extern char occaOpenMPDefines[" << mpChars  << "];" << std::endl
     << "    extern char occaOpenCLDefines[" << clChars  << "];" << std::endl
     << "    extern char occaCUDADefines["   << cuChars  << "];" << std::endl
     << "    extern char occaCOIDefines["   << coiChars << "];" << std::endl
     << "    extern char occaCOIMain["   << coiMainChars << "];" << std::endl
     << "}" << std::endl;

  fs.close();

  fs.open(occaKernelDefinesSource.c_str());

  fs << ns << std::endl
     << pt << std::endl
     << mp << std::endl
     << cl << std::endl
     << cu << std::endl
     << coi << std::endl
     << coiMain << std::endl
     << "}" << std::endl;

  fs.close();
}
