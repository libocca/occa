/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2017 David Medina and Tim Warburton
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

#include "occa.hpp"

occa::cli::command occaCommand;

bool runCache(const occa::cli::command &command,
              occa::jsonArray_t order,
              occa::jsonObject_t options,
              occa::jsonArray_t arguments);

bool runClear(const occa::cli::command &command,
              occa::jsonArray_t order,
              occa::jsonObject_t options,
              occa::jsonArray_t arguments);

bool runCompile(const occa::cli::command &command,
                occa::jsonArray_t order,
                occa::jsonObject_t options,
                occa::jsonArray_t arguments);

bool runEnv(const occa::cli::command &command,
            occa::jsonArray_t order,
            occa::jsonObject_t options,
            occa::jsonArray_t arguments);

bool runInfo(const occa::cli::command &command,
             occa::jsonArray_t order,
             occa::jsonObject_t options,
             occa::jsonArray_t arguments);

bool runBashAutocomplete(const occa::cli::command &command,
                         occa::jsonArray_t order,
                         occa::jsonObject_t options,
                         occa::jsonArray_t arguments);

int main(int argc, char **argv) {
  occa::cli::command cacheCommand;
  cacheCommand
    .withName("cache")
    .withCallback(runCache)
    .withDescription("Cache kernels")
    .addArgument("LIBRARY",
                 "Library where kernels will be cached under",
                 true)
    .addRepetitiveArgument("FILE",
                           "OKL files that will be cached.",
                           true);

  occa::cli::command clearCommand;
  clearCommand
    .withName("clear")
    .withCallback(runClear)
    .withDescription("Clears cached files and cache locks")
    .addOption(occa::cli::option('a', "all",
                                 "Clear cached kernels, cached libraries, and locks.")
               .stopsExpansion())
    .addOption(occa::cli::option('k', "kernel",
                                 "Clear cached kernel.")
               .reusable()
               .expandsFunction("ls ${OCCA_CACHE_DIR:-${HOME}/.occa}/cache"))
    .addOption(occa::cli::option('\0', "kernels",
                                 "Clear cached kernels."))
    .addOption(occa::cli::option('l', "lib",
                                 "Clear cached library.")
               .reusable()
               .expandsFunction("ls ${OCCA_CACHE_DIR:-${HOME}/.occa}/libraries"))
    .addOption(occa::cli::option('\0', "libraries",
                                 "Clear cached libraries."))
    .addOption(occa::cli::option('o', "locks",
                                 "Clear cache locks"))
    .addOption(occa::cli::option('y', "yes",
                                 "Automatically answer everything with [y/yes]"));

  occa::cli::command compileCommand;
  compileCommand
    .withName("compile")
    .withCallback(runCompile)
    .withDescription("Compile and cache kernels")
    .addRepetitiveArgument("RECIPE",
                           "JSON/JS recipe file. "
                           "The file should be an object with all device and kernel property combinations that will be compiled.",
                           true);

  occa::cli::command envCommand;
  envCommand
    .withName("env")
    .withCallback(runEnv)
    .withDescription("Print environment variables used in OCCA");

  occa::cli::command infoCommand;
  infoCommand
    .withName("info")
    .withCallback(runInfo)
    .withDescription("Prints information about available OCCA modes");

  occa::cli::command autocompleteBash;
  autocompleteBash
    .withName("bash")
    .withCallback(runBashAutocomplete)
    .withDescription("Prints bash functions to autocomplete occa commands and arguments");

  occa::cli::command autocompleteCommand;
  autocompleteCommand
    .withName("autocomplete")
    .withDescription("Prints shell functions to autocomplete occa commands and arguments")
    .requiresCommand()
    .addCommand(autocompleteBash);

  occaCommand
    .withDescription("Can be used to display information of cache kernels.")
    .requiresCommand()
    .addCommand(cacheCommand)
    .addCommand(clearCommand)
    .addCommand(envCommand)
    .addCommand(infoCommand)
    .addCommand(autocompleteCommand);

  occaCommand.run(argc, (const char**) argv);

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

bool removeDir(const std::string &dir, const bool promptCheck = true) {
  if (!occa::sys::dirExists(dir)) {
    return false;
  }

  std::string input;

  std::cout << "  Removing [" << dir << "]";
  if (promptCheck) {
    std::cout << ", are you sure? [y/n]:  ";
    std::cin >> input;
    occa::strip(input);
  } else {
    std::cout << '\n';
    input = "y";
  }

  if (input == "y") {
    std::string command = "rm -rf " + dir;
    occa::ignoreResult( system(command.c_str()) );
  } else if (input != "n") {
    std::cout << "  Input must be [y] or [n], ignoring clear command\n";
  }
  return true;
}

bool runClear(const occa::cli::command &command,
              occa::jsonArray_t order,
              occa::jsonObject_t options,
              occa::jsonArray_t arguments) {

  occa::cJsonObjectIterator it = options.begin();

  if (it == options.end()) {
    return false;
  }
  bool removedSomething = false;
  const bool promptCheck = (options.find("yes") == options.end());
  while (it != options.end()) {
    if (it->first == "all") {
      removedSomething |= removeDir(occa::env::OCCA_CACHE_DIR, promptCheck);
    } else if (it->first == "lib") {
      const occa::jsonArray_t &libGroups = it->second.array();
      for (int i = 0; i < (int) libGroups.size(); ++i) {
        const occa::jsonArray_t &libs = libGroups[i].array();
        for (int j = 0; j < (int) libs.size(); ++j) {
          removedSomething |= removeDir(occa::io::libraryPath() +
                                        libs[j].string(),
                                        promptCheck);
        }
      }
    } else if (it->first == "libraries") {
      removedSomething |= removeDir(occa::io::libraryPath(), promptCheck);
    } else if (it->first == "kernel") {
      const occa::jsonArray_t &kernelGroups = it->second.array();
      for (int i = 0; i < (int) kernelGroups.size(); ++i) {
        const occa::jsonArray_t &kernels = kernelGroups[i].array();
        for (int j = 0; j < (int) kernels.size(); ++j) {
          removedSomething |= removeDir(occa::io::cachePath() +
                                        kernels[j].string(),
                                        promptCheck);
        }
      }
    } else if (it->first == "kernels") {
      removedSomething |= removeDir(occa::io::cachePath(), promptCheck);
    } else if (it->first == "locks") {
      const std::string lockPath = occa::env::OCCA_CACHE_DIR + "locks/";
      removedSomething |= removeDir(lockPath, promptCheck);
    }
    ++it;
  }
  if (!removedSomething) {
    std::cout << "  Nothing to remove.\n";
  }
  return true;
}

bool runCache(const occa::cli::command &command,
              occa::jsonArray_t order,
              occa::jsonObject_t options,
              occa::jsonArray_t arguments) {
  std::string libDir = occa::io::libraryPath() + arguments[0].string() + "/";
  occa::sys::mkpath(libDir);

  const int fileCount = arguments.size();
  for (int i = 1; i < fileCount; ++i) {
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

bool runCompile(const occa::cli::command &command,
                occa::jsonArray_t order,
                occa::jsonObject_t options,
                occa::jsonArray_t arguments) {
  return true;
}

bool runEnv(const occa::cli::command &command,
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

bool runInfo(const occa::cli::command &command,
             occa::jsonArray_t order,
             occa::jsonObject_t options,
             occa::jsonArray_t arguments) {
  occa::printModeInfo();
  ::exit(0);
  return true;
}

bool runBashAutocomplete(const occa::cli::command &command,
                         occa::jsonArray_t order,
                         occa::jsonObject_t options,
                         occa::jsonArray_t arguments) {
  occaCommand.printBashAutocomplete();
  ::exit(0);
  return true;
}
