/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 David Medina and Tim Warburton
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

#include <occa.hpp>
#include <occa/lang/modes/serial.hpp>
#include <occa/lang/modes/openmp.hpp>
#include <occa/lang/modes/opencl.hpp>
#include <occa/lang/modes/cuda.hpp>

using namespace occa;

// Global to make it accessible to autocomplete
cli::command occaCommand;

std::string envEcho(const std::string &arg) {
  std::string ret = env::var(arg);
  return (ret.size() ? ret : "[NOT SET]");
}

template <class TM>
std::string envEcho(const std::string &arg, const TM &defaultValue) {
  std::string ret = env::var(arg);
  return (ret.size() ? ret : toString(defaultValue));
}

bool removeDir(const std::string &dir, const bool promptCheck = true) {
  if (!sys::dirExists(dir)) {
    return false;
  }

  std::string input;

  std::cout << "  Removing [" << dir << "]";
  if (promptCheck) {
    std::cout << ", are you sure? [y/n]:  ";
    std::cin >> input;
    strip(input);
  } else {
    std::cout << '\n';
    input = "y";
  }

  if (input == "y") {
    std::string command = "rm -rf " + dir;
    ignoreResult( system(command.c_str()) );
  } else if (input != "n") {
    std::cout << "  Input must be [y] or [n], ignoring clear command\n";
  }
  return true;
}

bool runClear(const cli::command &command,
              jsonArray order,
              jsonObject options,
              jsonArray arguments) {

  jsonObject::const_iterator it = options.begin();
  if (it == options.end()) {
    return false;
  }

  bool removedSomething = false;
  const bool promptCheck = (options.find("yes") == options.end());
  while (it != options.end()) {
    if (it->first == "all") {
      removedSomething |= removeDir(env::OCCA_CACHE_DIR, promptCheck);
    } else if (it->first == "lib") {
      const jsonArray &libGroups = it->second.array();
      for (int i = 0; i < (int) libGroups.size(); ++i) {
        const jsonArray &libs = libGroups[i].array();
        for (int j = 0; j < (int) libs.size(); ++j) {
          removedSomething |= removeDir(io::libraryPath() +
                                        (std::string) libs[j],
                                        promptCheck);
        }
      }
    } else if (it->first == "libraries") {
      removedSomething |= removeDir(io::libraryPath(), promptCheck);
    } else if (it->first == "kernels") {
      removedSomething |= removeDir(io::cachePath(), promptCheck);
    } else if (it->first == "locks") {
      const std::string lockPath = env::OCCA_CACHE_DIR + "locks/";
      removedSomething |= removeDir(lockPath, promptCheck);
    }
    ++it;
  }
  if (!removedSomething) {
    std::cout << "  Nothing to remove.\n";
  }
  return true;
}

bool runVersion(const cli::command &command,
                jsonArray order,
                jsonObject options,
                jsonArray arguments) {

  jsonObject::const_iterator it = options.begin();
  if (it == options.end()) {
    std::cout << OCCA_VERSION_STR << '\n';
  }
  else if (options.find("okl") != options.end()) {
    std::cout << OKL_VERSION_STR << '\n';
  }
  return true;
}

bool runCache(const cli::command &command,
              jsonArray order,
              jsonObject options,
              jsonArray arguments) {
  std::string libDir = (io::libraryPath()
                        + (std::string) arguments[0]
                        + "/");
  sys::mkpath(libDir);

  const int fileCount = arguments.size();
  for (int i = 1; i < fileCount; ++i) {
    const std::string srcFile = arguments[i];
    const std::string destFile = libDir + io::basename(srcFile);

    if (!sys::fileExists(srcFile)) {
      std::cerr << yellow("Warning") << ": File '"
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

bool runTranslate(const cli::command &command,
                  jsonArray order,
                  jsonObject options,
                  jsonArray arguments) {

  const std::string mode = options["mode"][0][0];
  const std::string filename = arguments[0];

  if (!io::exists(filename)) {
    occa::printError("File [" + filename + "] doesn't exist" );
    ::exit(1);
  }

  properties props;
  json &propsList = options["props"];
  const int propsCount = propsList.size();
  for (int i = 0; i < propsCount; ++i) {
    props += occa::properties((std::string) propsList[i][0]);
  }

  lang::parser_t *parser = NULL;
  if (mode == "Serial") {
    parser = new lang::okl::serialParser(props);
  } else if (mode == "OpenMP") {
    parser = new lang::okl::openmpParser(props);
  } else if (mode == "OpenCL") {
    parser = new lang::okl::openclParser(props);
  } else if (mode == "CUDA") {
    parser = new lang::okl::cudaParser(props);
  }

  if (!parser) {
    occa::printError("Unable to translate for mode [" + mode + "]");
    ::exit(1);
  }

  parser->parseFile(filename);

  bool success = parser->succeeded();
  if (success) {
    std::cout << parser->toString();
  }
  delete parser;
  return success;
}

bool runEnv(const cli::command &command,
            jsonArray order,
            jsonObject options,
            jsonArray arguments) {

  std::cout << "  Basic:\n"
            << "    - OCCA_DIR                   : " << envEcho("OCCA_DIR") << "\n"
            << "    - OCCA_CACHE_DIR             : " << envEcho("OCCA_CACHE_DIR") << "\n"
            << "    - OCCA_VERBOSE               : " << envEcho("OCCA_VERBOSE") << "\n"
            << "    - OCCA_UNSAFE                : " << OCCA_UNSAFE << "\n"

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
            << "    - OCCA_INCLUDE_PATH          : " << envEcho("OCCA_INCLUDE_PATH") << "\n"
            << "    - OCCA_LIBRARY_PATH          : " << envEcho("OCCA_LIBRARY_PATH") << "\n"
            << "    - OCCA_OPENCL_COMPILER_FLAGS : " << envEcho("OCCA_OPENCL_COMPILER_FLAGS") << "\n"
            << "    - OCCA_CUDA_COMPILER         : " << envEcho("OCCA_CUDA_COMPILER") << "\n"
            << "    - OCCA_CUDA_COMPILER_FLAGS   : " << envEcho("OCCA_CUDA_COMPILER_FLAGS") << "\n";
  return true;
}

bool runInfo(const cli::command &command,
             jsonArray order,
             jsonObject options,
             jsonArray arguments) {
  printModeInfo();
  return true;
}

bool runModes(const cli::command &command,
              jsonArray order,
              jsonObject options,
              jsonArray arguments) {

  strToModeMap &modes = modeMap();
  strToModeMap::iterator it = modes.begin();
  while (it != modes.end()) {
    std::cout << it->first << '\n';
    ++it;
  }
  return true;
}

bool runBashAutocomplete(const cli::command &command,
                         jsonArray order,
                         jsonObject options,
                         jsonArray arguments) {
  occaCommand.printBashAutocomplete();
  return true;
}

int main(const int argc, const char **argv) {
  cli::command versionCommand;
  versionCommand
    .withName("version")
    .withCallback(runVersion)
    .withDescription("Prints OCCA library version")
    .addOption(cli::option("okl",
                           "Print the OKL language version")
               .stopsExpansion());

  cli::command cacheCommand;
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

  cli::command clearCommand;
  clearCommand
    .withName("clear")
    .withCallback(runClear)
    .withDescription("Clears cached files and cache locks")
    .addOption(cli::option('a', "all",
                           "Clear cached kernels, cached libraries, and locks.")
               .stopsExpansion())
    .addOption(cli::option("kernels",
                           "Clear cached kernels."))
    .addOption(cli::option('l', "lib",
                           "Clear cached library.")
               .reusable()
               .expandsFunction("ls ${OCCA_CACHE_DIR:-${HOME}/.occa}/libraries"))
    .addOption(cli::option("libraries",
                           "Clear cached libraries."))
    .addOption(cli::option('o', "locks",
                           "Clear cache locks"))
    .addOption(cli::option('y', "yes",
                           "Automatically answer everything with [y/yes]"));

  cli::command translateCommand;
  translateCommand
    .withName("translate")
    .withCallback(runTranslate)
    .withDescription("Translate kernels")
    .addOption(cli::option('m', "mode",
                           "Output mode")
               .isRequired()
               .withArgs(1)
               .expandsFunction("occa modes"))
    .addOption(cli::option('p', "props",
                           "Kernel properties")
               .reusable()
               .withArgs(1))
    .addArgument("FILE",
                 "An .okl file",
                 true);

  cli::command envCommand;
  envCommand
    .withName("env")
    .withCallback(runEnv)
    .withDescription("Print environment variables used in OCCA");

  cli::command infoCommand;
  infoCommand
    .withName("info")
    .withCallback(runInfo)
    .withDescription("Prints information about available backend modes");

  cli::command modesCommand;
  modesCommand
    .withName("modes")
    .withCallback(runModes)
    .withDescription("Prints available backend modes");

  cli::command autocompleteBash;
  autocompleteBash
    .withName("bash")
    .withCallback(runBashAutocomplete)
    .withDescription("Prints bash functions to autocomplete occa commands and arguments");

  cli::command autocompleteCommand;
  autocompleteCommand
    .withName("autocomplete")
    .withDescription("Prints shell functions to autocomplete occa commands and arguments")
    .requiresCommand()
    .addCommand(autocompleteBash);

  occaCommand
    .withDescription("Can be used to display information of cache kernels.")
    .requiresCommand()
    .addCommand(versionCommand)
    .addCommand(cacheCommand)
    .addCommand(clearCommand)
    .addCommand(translateCommand)
    .addCommand(envCommand)
    .addCommand(infoCommand)
    .addCommand(modesCommand)
    .addCommand(autocompleteCommand)
    .run(argc, (const char**) argv);

  return 0;
}
