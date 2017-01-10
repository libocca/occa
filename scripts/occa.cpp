/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2016 David Medina and Tim Warburton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */

#include <fstream>

#include "occa/base.hpp"
#include "occa/parser/tools.hpp"
#include "occa/tools/env.hpp"
#include "occa/tools/io.hpp"
#include "occa/tools/misc.hpp"
#include "occa/tools/sys.hpp"

void printHelp();

void runHelp(const std::string &cmd);

void runClearOn(const std::string &path);
void runClearCache(const int argc, std::string *args);
void runClearLocks();

void runEnv();

void runInfo();

void runUpdate(const int argc, std::string *args);

int main(int argc, char **argv) {
  --argc;

  if (argc == 0) {
    printHelp();
    return 0;
  }

  std::string *args = new std::string[argc];
  for (int i = 0; i < argc; ++i) {
    args[i] = argv[i + 1];
  }

  if (args[0] == "help") {
    if (1 < argc) {
      runHelp(args[1]);
    } else {
      printHelp();
    }
  }
  else if (args[0] == "clear") {
    if (1 < argc) {
      if (args[1] == "all") {
        runClearOn(occa::env::OCCA_CACHE_DIR);
      } else if (args[1] == "cache") {
        runClearCache(argc - 2, args + 2);
      } else if (args[1] == "locks") {
        runClearLocks();
      } else {
        printHelp();
      }
    }
    else {
      printHelp();
    }
  } else if (args[0] == "env") {
    runEnv();
  } else if (args[0] == "info") {
    runInfo();
  } else if (args[0] == "update") {
    --argc;

    if (argc < 2) {
      runHelp("update");
    } else {
      runUpdate(argc, args + 1);
    }
  } else {
    printHelp();
  }

  delete [] args;
  return 0;
}

void printHelp() {
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

void runHelp(const std::string &cmd) {
  if (cmd == "cache") {
    std::cout << "- OCCA caches kernels in:\n\n"

              << "      $OCCA_CACHE_DIR/\n\n"

              << "  which defaults to:\n\n"

              << "      ~/.occa/\n\n"

              << "- Kernel libraries are represented in kernels as:\n\n"

              << "      occa://libraryName/kernel.okl\n\n"

              << "  which can be found in:\n\n"

              << "      $OCCA_CACHE_DIR/libraries/libraryName/kernel.okl\n\n"

              << "- If a kernel is not in a library, the intermediate files\n"
              << "  and binaries can still be found in:\n\n"

              << "      $OCCA_CACHE_DIR/cache/<hash>/source.occa\n"
              << "      $OCCA_CACHE_DIR/cache/<hash>/binary\n\n"

              << "  where the <hash> is printed (by default) when the kernel\n"
              << "  is built\n";
  } else if (cmd == "clear") {
    std::cout << "  Clears kernels that were cached and compilation locks\n"
              << "    - occa clear all\n"
              << "    - occa clear cache library1[, library2, ...]\n"
              << "    - occa clear locks\n";
  } else if (cmd == "env") {
    std::cout << "  The following are optional environment variables and their use\n"
              << "  Basic:\n"
              << "    - OCCA_CACHE_DIR: Directory where kernels and their compiled binaries are cached\n"
              << "                        [Defaults to ~/.occa]\n\n"

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
  } else if (cmd == "info") {
    std::cout << "  Prints available devices for enabled backends\n"
              << "  For example:\n"
              << "    - Basic CPU information\n"
              << "    - OpenCL platforms and devices\n"
              << "    - CUDA devices\n";
  } else if (cmd == "update") {
    std::cout << "  Updates the library or application's kernel files in a cache directory\n"
              << "  This is used to find kernels at run-time without specifying an absolute path to the files\n"
              << "  See [occa help env] for more information about environment variables\n";
  }
}

std::string envEcho(const std::string &arg) {
  std::string ret = occa::env::var(arg);
  return (ret.size() ? ret : "[NOT SET]");
}

void runClearOn(const std::string &path) {
  std::string input;

  std::cout << "  Removing [" << path << "*], are you sure? [y/n]:  ";
  std::cin >> input;
  occa::strip(input);

  if (input == "y") {
    std::string command = "rm -rf " + path + "*";
    occa::ignoreResult( system(command.c_str()) );
  }
  else if (input != "n") {
    std::cout << "  Input must be [y] or [n], ignoring clear command\n";
  }
}

void runClearCache(const int argc, std::string *args) {
  const std::string cPath = occa::io::cachePath();
  const std::string lPath = occa::io::libraryPath();

  const bool cPathExists = occa::sys::fileExists(cPath);
  const bool lPathExists = occa::sys::fileExists(lPath);

  if (!lPathExists && !cPathExists) {
    std::cout << "  Cache is already empty\n";
    return;
  }

  if (argc == 0) {
    if (cPathExists) {
      runClearOn(cPath);
    }
    if (lPathExists) {
      runClearOn(lPath);
    }
  }
  else {
    for (int i = 0; i < argc; ++i) {
      const std::string argLibPath = lPath + args[i] + "/";

      if (occa::sys::fileExists(argLibPath)) {
        runClearOn(argLibPath);
      } else {
        std::cout << "  Cache for [" << args[i] << "] is already empty\n";
      }
    }
  }
}

void runClearLocks() {
  const std::string lockPath = occa::env::OCCA_CACHE_DIR + "locks/";
  if (occa::sys::fileExists(lockPath)) {
    runClearOn(lockPath);
  } else {
    std::cout << "  No locks found\n";
  }
}

void runEnv() {
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

void runInfo() {
  occa::printModeInfo();
}

void runUpdate(const int argc, std::string *args) {
  std::string &library = args[0];
  std::string libDir   = occa::io::dirname("occa://" + library + "/");

  occa::sys::mkpath(libDir);

  for (int i = 1; i < argc; ++i) {
    std::string originalFile = occa::io::filename(args[i], true);

    if (!occa::sys::fileExists(originalFile)) {
      continue;
    }

    std::string filename = occa::io::basename(originalFile);
    std::string newFile  = libDir + filename;

    std::ifstream originalS(originalFile.c_str(), std::ios::binary);
    std::ofstream newS(     newFile.c_str()     , std::ios::binary);

    newS << originalS.rdbuf();

    originalS.close();
    newS.close();
  }
}
