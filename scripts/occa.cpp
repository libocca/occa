#include "occa.hpp"

void printHelp();

void runHelp(const std::string &cmd);

void runClearOn(const std::string &path);
void runClearCache(const int argc, std::string *args);
void runClearLocks();

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
  else if(args[0] == "clear"){
    if(1 < argc){
      if(args[1] == "cache")
        runClearCache(argc - 2, args + 2);
      else if(args[1] == "locks")
        runClearLocks();
      else
        printHelp();
    }
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
            << "  Otherwise run a command with:          [occa <command> <arguments>]\n\n"

            << "  Available commands:\n"
            << "    - clear\n"
            << "    - env\n"
            << "    - info\n"
            << "    - update <library or application name> <file>[, file2...]\n\n"

            << "  Additional information:\n"
            << "    - occa help cache\n";
}

void runHelp(const std::string &cmd){
  if(cmd == "cache"){
    std::cout << "- OCCA caches kernels in:\n\n"

              << "      $OCCA_CACHE_DIR/\n\n"

              << "  which defaults to:\n\n"

              << "      ~/._occa/\n\n"

              << "- Kernel libraries are represented in kernels as:\n\n"

              << "      [libraryName]/kernel.okl\n\n"

              << "  which can be found in:\n\n"

              << "      $OCCA_CACHE_DIR/libraries/libraryName/kernel.okl\n\n"

              << "- If a kernel is not in a library, the intermediate files\n"
              << "  and binaries can still be found in:\n\n"

              << "      $OCCA_CACHE_DIR/kernels/<hash>/source.occa\n"
              << "      $OCCA_CACHE_DIR/kernels/<hash>/binary\n\n"

              << "  where the <hash> is printed (by default) when the kernel\n"
              << "  is built\n";
  }
  else if(cmd == "clear"){
    std::cout << "  Clears kernels that were cached and compilation locks\n"
              << "    - occa clear cache <library1, library2, ...>\n"
              << "    - occa clear locks\n";
  }
  else if(cmd == "env"){
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
  std::string ret = occa::env::var(arg);

  return (ret.size() ? ret : "[NOT SET]");
}

void runClearOn(const std::string &path){
  std::string input;

  std::cout << "  Removing [" << path << "*], are you sure? [y/n]:  ";
  std::cin >> input;

  occa::strip(input);

  if(input == "y"){
    std::string command = "rm -rf " + path + "*";
    system(command.c_str());
  }
  else if(input != "n")
    std::cout << "  Input must be [y] or [n], ignoring clear command\n";
}

void runClearCache(const int argc, std::string *args){
  const std::string libPath = occa::env::OCCA_CACHE_DIR + "libraries/";
  const std::string isoPath = occa::env::OCCA_CACHE_DIR + "kernels/";

  const bool libPathExists = occa::sys::fileExists(libPath);
  const bool isoPathExists = occa::sys::fileExists(isoPath);

  if(!libPathExists && !isoPathExists){
    std::cout << "  Cache is already empty\n";
    return;
  }

  if(argc == 0){
    if(libPathExists)
      runClearOn(libPath);
    if(isoPathExists)
      runClearOn(isoPath);
  }
  else {
    for(int i = 0; i < argc; ++i){
      const std::string argLibPath = libPath + args[i] + "/";

      if(occa::sys::fileExists(argLibPath))
        runClearOn(argLibPath);
      else
        std::cout << "  Cache for [" << args[i] << "] is already empty\n";
    }
  }
}

void runClearLocks(){
  const std::string lockPath = occa::env::OCCA_CACHE_DIR + "locks/";

  if(occa::sys::fileExists(lockPath))
    runClearOn(lockPath);
  else
    std::cout << "  No locks found\n";
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
