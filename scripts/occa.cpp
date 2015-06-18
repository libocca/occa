#include "occa.hpp"

void printHelp();

void runHelp(const std::string &cmd);
void runEnv();
void runInfo();
void runUpdate(const int argc, std::string *args);

int main(int argc, char **argv){
  --argc;

  if(argc == 0){
    printHelp();
    return 0;
  }

  std::string *args = new std::string[argc];

  for(int i = 0; i < argc; ++i)
    args[i] = argv[i + 1];

  if(args[0] == "help"){
    if(1 < argc)
      runHelp(args[1]);
    else
      printHelp();
  }
  else if(args[0] == "env"){
    runEnv();
  }
  else if(args[0] == "info"){
    runInfo();
  }
  else if(args[0] == "update"){
    --argc;

    if(argc < 2)
      runHelp("update");
    else
      runUpdate(argc, args + 1);
  }
  else
    printHelp();

  delete [] args;

  return 0;
}

void printHelp(){
  std::cout << "  For help on a specific command, type:  [occa help <command>]\n"
            << "  Otherwise run a command with:          [occa <command> <arguments>]\n"
            << "  Available commands:\n"
            << "    - env\n"
            << "    - info\n"
            << "    - update <library or application name> <file>[, file2...]\n";
}

void runHelp(const std::string &cmd){
  if(cmd == "env"){
    std::cout << "  The following are optional environment variables and their use\n"
              << "  Basic:\n"
              << "    - OCCA_CACHE_DIR: Directory where kernels and their compiled binaries are cached\n"
              << "                        [Defaults to ~/._occa]\n\n"

              << "  Makefile:\n"
              << "    - CXX           : C++ compiler\n"
              << "                        [Defaults to g++     ]\n"
              << "    - CXXFLAGS      : C++ compiler flags\n"
              << "                        [Defaults to -g      ]\n"
              << "    - FC            : Fortran compiler\n"
              << "                        [Defaults to gfortran]\n"
              << "    - FCFLAGS       : Fortran compiler flags\n"
              << "                        [Defaults to -g      ]\n"
              << "    - LDFLAGS       : Extra linking flags\n"
              << "                        [Defaults to \"\"      ]\n\n"

              << "  Backend Support:\n"
              << "  [WARNING] These are auto-detected, manually setting these will require \n"
              << "              the user to explicitly put library directories and libraries through\n"
              << "              CXXFLAGS and LDFLAGS (for example: -L/path/to/CL -lopencl)\n\n"

              << "    - OCCA_OPENMP_ENABLED: Set to 0 if you wish to manually disable it\n"
              << "    - OCCA_OPENCL_ENABLED: Set to 0 if you wish to manually disable it\n"
              << "    - OCCA_CUDA_ENABLED  : Set to 0 if you wish to manually disable it\n"
              << "    - OCCA_COI_ENABLED   : Set to 0 if you wish to manually disable it\n\n"

              << "  Run-Time Options:\n"
              << "    - OCCA_CXX                   : C++ compiler used on the kernels\n"
              << "                                      [Defaults to g++             ]\n"
              << "    - OCCA_CXXFLAGS              : C++ compiler used on the kernels\n"
              << "                                      [Defaults to -g              ]\n"
              << "    - OCCA_OPENCL_COMPILER_FLAGS : Compiler flags used when compiling OpenCL kernels\n"
              << "                                      [Defaults to -cl-opt-disable ]\n"
              << "    - OCCA_CUDA_COMPILER         : Compiler used when compiling CUDA kernels\n"
              << "                                      [Defaults to nvcc            ]\n"
              << "    - OCCA_CUDA_COMPILER_FLAGS   : Compiler flags used when compiling CUDA kernels\n"
              << "                                      [Defaults to -g              ]\n";
  }
  else if(cmd == "info"){
    std::cout << "  Prints available devices for enabled backends\n"
              << "  For example:\n"
              << "    - Basic CPU information\n"
              << "    - OpenCL platforms and devices\n"
              << "    - CUDA devices\n";
  }
  else if(cmd == "update"){
    std::cout << "  Updates the library or application's kernel files in a cache directory\n"
              << "  This is used to find kernels at run-time without specifying an absolute path to the files\n"
              << "  See [occa help env] for more information about environment variables\n";
  }
}

std::string envEcho(const std::string &arg){
  std::string ret = occa::sys::echo(arg);

  return (ret.size() ? ret : "[NOT SET]");
}

void runEnv(){
  std::cout << "  The following are optional environment variables and their values\n"
            << "  Basic:\n"
            << "    - OCCA_CACHE_DIR             : " << envEcho("OCCA_CACHE_DIR") << "\n"
            << "  Makefile:\n"
            << "    - CXX                        : " << envEcho("CXX") << "\n"
            << "    - CXXFLAGS                   : " << envEcho("CXXFLAGS") << "\n"
            << "    - FC                         : " << envEcho("FC") << "\n"
            << "    - FCFLAGS                    : " << envEcho("FCFLAGS") << "\n"
            << "    - LDFLAGS                    : " << envEcho("LDFLAGS") << "\n"

            << "  Backend Support:\n"
            << "    - OCCA_OPENMP_ENABLED        : " << envEcho("OCCA_OPENMP_ENABLED") << "\n"
            << "    - OCCA_OPENCL_ENABLED        : " << envEcho("OCCA_OPENCL_ENABLED") << "\n"
            << "    - OCCA_CUDA_ENABLED          : " << envEcho("OCCA_CUDA_ENABLED") << "\n"
            << "    - OCCA_COI_ENABLED           : " << envEcho("OCCA_COI_ENABLED") << "\n"

            << "  Run-Time Options:\n"
            << "    - OCCA_CXX                   : " << envEcho("OCCA_CXX") << "\n"
            << "    - OCCA_CXXFLAGS              : " << envEcho("OCCA_CXXFLAGS") << "\n"
            << "    - OCCA_OPENCL_COMPILER_FLAGS : " << envEcho("OCCA_OPENCL_COMPILER_FLAGS") << "\n"
            << "    - OCCA_CUDA_COMPILER         : " << envEcho("OCCA_CUDA_COMPILER") << "\n"
            << "    - OCCA_CUDA_COMPILER_FLAGS   : " << envEcho("OCCA_CUDA_COMPILER_FLAGS") << "\n";
}

void runInfo(){
  occa::printAvailableDevices();
}

void runUpdate(const int argc, std::string *args){
  std::string &library = args[0];
  std::string libDir   = occa::sys::getFilename("[" + library + "]") + "/";

  occa::sys::mkpath(libDir);

  for(int i = 1; i < argc; ++i){
    std::string originalFile = occa::sys::getFilename(args[i]);

    if(!occa::sys::fileExists(originalFile))
      continue;

    std::string filename = occa::getOnlyFilename(originalFile);
    std::string newFile  = libDir + filename;

    std::ifstream originalS(originalFile.c_str(), std::ios::binary);
    std::ofstream newS(     newFile.c_str()     , std::ios::binary);

    newS << originalS.rdbuf();

    originalS.close();
    newS.close();
  }
}
