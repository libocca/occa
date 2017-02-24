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
#include "occa/tools/args.hpp"
#include "occa/tools/env.hpp"
#include "occa/tools/io.hpp"
#include "occa/tools/json.hpp"
#include "occa/tools/misc.hpp"
#include "occa/tools/sys.hpp"

bool runCache(const occa::args::command &command,
              occa::jsonArray_t order,
              occa::jsonObject_t options,
              occa::jsonArray_t arguments);
bool runClear(const occa::args::command &command,
              occa::jsonArray_t order,
              occa::jsonObject_t options,
              occa::jsonArray_t arguments);
bool runCompile(const occa::args::command &command,
                occa::jsonArray_t order,
                occa::jsonObject_t options,
                occa::jsonArray_t arguments);
bool runEnv(const occa::args::command &command,
            occa::jsonArray_t order,
            occa::jsonObject_t options,
            occa::jsonArray_t arguments);
bool runInfo(const occa::args::command &command,
             occa::jsonArray_t order,
             occa::jsonObject_t options,
             occa::jsonArray_t arguments);

int main(int argc, char **argv) {
  occa::args::command mainCommand;
  occa::args::command cacheCommand, clearCommand, compileCommand, envCommand, infoCommand;

  mainCommand
    .withDescription("Can be used to display information of cache kernels.");

  cacheCommand
    .withName("cache")
    .withCallback(runCache)
    .withDescription("Cache kernels")
    .addOption('l', "lib",
               "Cache kernels in this library directory",
               1, true)
    .addRepetitiveArgument("FILE",
                           "OKL files that will be cached.",
                           true);

  clearCommand
    .withName("clear")
    .withCallback(runClear)
    .withDescription("Clears cached files and cache locks")
    .addOption('a', "all",
               "Clear cached kernels, cached libraries, and locks.")
    .addOption('k', "kernels",
               "Clear cached kernels.")
    .addOption('l', "lib",
               "Clear cached library.", 1)
    .addOption('\0', "libraries",
               "Clear cached libraries.")
    .addOption('o', "locks",
               "Clear cache locks");

  compileCommand
    .withName("compile")
    .withCallback(runCompile)
    .withDescription("Compile and cache kernels")
    .addRepetitiveArgument("RECIPE",
                           "JSON/JS recipe file. "
                           "The file should be an object with all device and kernel property combinations that will be compiled.",
                           true);

  envCommand
    .withName("env")
    .withCallback(runEnv)
    .withDescription("Print environment variables used in OCCA");

  infoCommand
    .withName("info")
    .withCallback(runInfo)
    .withDescription("Prints information about available OCCA modes");

  mainCommand
    .requiresCommand()
    .addCommand(cacheCommand)
    .addCommand(clearCommand)
    .addCommand(envCommand)
    .addCommand(infoCommand);

  mainCommand.run(argc, (const char**) argv);

  return 0;
}

std::string envEcho(const std::string &arg) {
  std::string ret = occa::env::var(arg);
  return (ret.size() ? ret : "[NOT SET]");
}

template <class TM>
std::string envEcho(const std::string &arg, const TM &defaultValue) {
  std::string ret = occa::env::var(arg);
  return (ret.size() ? ret : occa::toString(defaultValue));
}

bool removeDir(const std::string &dir) {
  const std::string fulldir = occa::io::endWithSlash(dir);

  if (!occa::sys::fileExists(fulldir)) {
    return false;
  }

  std::string input;

  std::cout << "  Removing [" << fulldir << "*], are you sure? [y/n]:  ";
  std::cin >> input;
  occa::strip(input);

  if (input == "y") {
    std::string command = "rm -rf " + fulldir + "*";
    occa::ignoreResult( system(command.c_str()) );
  } else if (input != "n") {
    std::cout << "  Input must be [y] or [n], ignoring clear command\n";
  }
  return true;
}

bool runClear(const occa::args::command &command,
              occa::jsonArray_t order,
              occa::jsonObject_t options,
              occa::jsonArray_t arguments) {

  occa::cJsonObjectIterator it = options.begin();

  if (it == options.end()) {
    return false;
  }
  bool removedSomething = false;
  while (it != options.end()) {
    if (it->first == "all") {
      removedSomething |= removeDir(occa::env::OCCA_CACHE_DIR);
    } else if (it->first == "lib") {
      const occa::jsonArray_t &libraries = it->second.array();
      for (int i = 0; i < (int) libraries.size(); ++i) {
        removedSomething |= removeDir(occa::io::libraryPath() +
                                      libraries[i].array()[0].string());
      }
    } else if (it->first == "libraries") {
      removedSomething |= removeDir(occa::io::libraryPath());
    } else if (it->first == "kernels") {
      removedSomething |= removeDir(occa::io::cachePath());
    } else if (it->first == "locks") {
      const std::string lockPath = occa::env::OCCA_CACHE_DIR + "locks/";
      removedSomething |= removeDir(lockPath);
    }
    ++it;
  }
  if (!removedSomething) {
    std::cout << "  Nothing to remove.\n";
  }
  return true;
}

bool runCache(const occa::args::command &command,
              occa::jsonArray_t order,
              occa::jsonObject_t options,
              occa::jsonArray_t arguments) {
  std::string libDir = occa::io::dirname("occa://" + options["lib"][0][0].string() + "/");
  occa::sys::mkpath(libDir);

  const int fileCount = arguments.size();
  for (int i = 0; i < fileCount; ++i) {
    const std::string &srcFile = arguments[i].string();
    const std::string destFile = libDir + occa::io::basename(srcFile);

    if (!occa::sys::fileExists(srcFile)) {
      std::cerr << occa::yellow("Warning") << ": File '"
                << srcFile << "' does not exist\n";
      continue;
    }

    std::ifstream src(srcFile.c_str(), std::ios::binary);
    std::ofstream dest(destFile.c_str(), std::ios::binary);

    dest << src.rdbuf();

    src.close();
    dest.close();
  }
  return true;
}

bool runCompile(const occa::args::command &command,
                occa::jsonArray_t order,
                occa::jsonObject_t options,
                occa::jsonArray_t arguments) {
  return true;
}

bool runEnv(const occa::args::command &command,
            occa::jsonArray_t order,
            occa::jsonObject_t options,
            occa::jsonArray_t arguments) {
  std::cout << "  Basic:\n"
            << "    - OCCA_DIR                   : " << envEcho("OCCA_DIR") << "\n"
            << "    - OCCA_CACHE_DIR             : " << envEcho("OCCA_CACHE_DIR") << "\n"
            << "    - OCCA_PATH                  : " << envEcho("OCCA_PATH") << "\n"

            << "  Makefile:\n"
            << "    - CXX                        : " << envEcho("CXX") << "\n"
            << "    - CXXFLAGS                   : " << envEcho("CXXFLAGS") << "\n"
            << "    - FC                         : " << envEcho("FC") << "\n"
            << "    - FCFLAGS                    : " << envEcho("FCFLAGS") << "\n"
            << "    - LDFLAGS                    : " << envEcho("LDFLAGS") << "\n"

            << "  Backend Support:\n"
            << "    - OCCA_OPENMP_ENABLED        : " << envEcho("OCCA_OPENMP_ENABLED", OCCA_OPENMP_ENABLED) << "\n"
            << "    - OCCA_OPENCL_ENABLED        : " << envEcho("OCCA_OPENCL_ENABLED", OCCA_OPENCL_ENABLED) << "\n"
            << "    - OCCA_CUDA_ENABLED          : " << envEcho("OCCA_CUDA_ENABLED", OCCA_CUDA_ENABLED) << "\n"

            << "  Run-Time Options:\n"
            << "    - OCCA_CXX                   : " << envEcho("OCCA_CXX") << "\n"
            << "    - OCCA_CXXFLAGS              : " << envEcho("OCCA_CXXFLAGS") << "\n"
            << "    - OCCA_OPENCL_COMPILER_FLAGS : " << envEcho("OCCA_OPENCL_COMPILER_FLAGS") << "\n"
            << "    - OCCA_CUDA_COMPILER         : " << envEcho("OCCA_CUDA_COMPILER") << "\n"
            << "    - OCCA_CUDA_COMPILER_FLAGS   : " << envEcho("OCCA_CUDA_COMPILER_FLAGS") << "\n";
  ::exit(0);
  return true;
}

bool runInfo(const occa::args::command &command,
             occa::jsonArray_t order,
             occa::jsonObject_t options,
             occa::jsonArray_t arguments) {
  occa::printModeInfo();
  ::exit(0);
  return true;
}
