#include <fstream>
#include <functional>
#include <stdexcept>

#include <occa/core.hpp>
#include <occa/internal/bin/occa.hpp>
#include <occa/internal/utils/env.hpp>
#include <occa/internal/io.hpp>
#include <occa/internal/utils/sys.hpp>

#include <occa/internal/lang/modes/serial.hpp>
#include <occa/internal/lang/modes/openmp.hpp>
#include <occa/internal/lang/modes/cuda.hpp>
#include <occa/internal/lang/modes/hip.hpp>
#include <occa/internal/lang/modes/opencl.hpp>
#include <occa/internal/lang/modes/metal.hpp>
#include <occa/internal/lang/modes/dpcpp.hpp>
#include <occa/internal/modes.hpp>
#include <occa/internal/modes.hpp>

#if BUILD_WITH_CLANG_BASED_TRANSPILER
#include <occa/internal/utils/transpiler_utils.h>
#endif
#include <memory>


namespace occa {
  namespace bin {
    std::string envEcho(const std::string &arg) {
      std::string ret = env::var(arg);
      return (ret.size() ? ret : "[NOT SET]");
    }

    template <class TM>
    std::string envEcho(const std::string &arg, const TM &defaultValue) {
      std::string ret = env::var(arg);
      return (ret.size() ? ret : toString(defaultValue));
    }

    bool safeRmrf(const std::string &dir, const bool promptCheck = true) {
      if (!io::isDir(dir) && !io::isFile(dir)) {
        return false;
      }

      std::string input;
      io::stdout << "  Removing [" << dir << "]";
      if (promptCheck) {
        io::stdout << ", are you sure? [y/n]:  ";
        std::cin >> input;
        strip(input);
      } else {
        io::stdout << '\n';
        input = "y";
      }

      if (input == "y") {
        sys::rmrf(dir);
      } else if (input != "n") {
        io::stdout << "  Input must be [y] or [n], ignoring clear command\n";
      }
      return true;
    }

    json getOptionProperties(const json &opt) {
      json props;
      for (int i = 0; i < opt.size(); ++i) {
        props += json((std::string) opt[i]);
      }
      return props;
    }

    json getOptionDefines(const json &opt) {
      json defines(json::object_);
      for (int i = 0; i < opt.size(); ++i) {
        const std::string defineStr = opt[i];
        if (!defineStr.size()) {
          continue;
        }

        strVector parts = split(defineStr, '=', '\\');
        const int partCount = (int) parts.size();
        const std::string name = parts[0];

        // Empty define
        if (partCount == 1) {
          defines[name] = "";
        }
        if (partCount < 2) {
          continue;
        }
        parts.erase(parts.begin());
        defines[name] = join(parts, "=");
      }
      return defines;
    }

    void printRemovedMessage(const bool removedSomething) {
      if (!removedSomething) {
        io::stdout << "  Nothing to remove.\n";
      }
    }

    bool runClear(const json &args) {
      const json &options = args["options"];

      const bool promptCheck = !options["yes"];

      if (options["all"] &&
          safeRmrf(env::OCCA_CACHE_DIR, promptCheck)) {
        printRemovedMessage(true);
        return true;
      }

      bool removedSomething = false;
      if (options["kernels"]) {
        removedSomething |= safeRmrf(io::cachePath(), promptCheck);
      }
      if (options["locks"]) {
        const std::string lockPath = env::OCCA_CACHE_DIR + "locks/";
        removedSomething |= safeRmrf(lockPath, promptCheck);
      }

      printRemovedMessage(removedSomething);

      return true;
    }

    bool runVersion(const json &args) {
      if (args["options/okl"]) {
        io::stdout << OKL_VERSION_STR << '\n';
      } else {
        io::stdout << OCCA_VERSION_STR << '\n';
      }
      return true;
    }

    int getTranspilerVersion(const json &options) {
      json jsonTranspileVersion = options["transpiler-version"];
      int transpilerVersion = 2;
      //INFO: have no idea why json here has array type
      if(!jsonTranspileVersion.isArray()) {
        return transpilerVersion;
      }
      json elem = jsonTranspileVersion.asArray()[0];
      if(!elem.isString()) {
        return transpilerVersion;
      }

      try {
        transpilerVersion = std::stoi(elem.string());
      } catch(const std::exception &)
      {
        return transpilerVersion;
      }
      return transpilerVersion;
    }

    namespace v2 {
        bool runTranspiler(const json &options,
                           const json &arguments,
                           const json &kernelProps,
                           const std::string &originalMode,
                           const std::string &mode)
        {
          using ParserBuildFunc = std::function<std::unique_ptr<lang::parser_t>(const json &params)>;
          static const std::map<std::string, ParserBuildFunc> originalParserBackends =
              {
                  {"", [](const json &params) {
                       return std::make_unique<lang::okl::serialParser>(params);
                   }},
                  {"serial", [](const json &params) {
                       return std::make_unique<lang::okl::serialParser>(params);
                   }},
                  {"openmp", [](const json &params) {
                       return std::make_unique<lang::okl::openmpParser>(params);
                   }},
                  {"cuda", [](const json &params) {
                       return std::make_unique<lang::okl::cudaParser>(params);
                   }},
                  {"hip", [](const json &params) {
                       return std::make_unique<lang::okl::hipParser>(params);
                   }},
                  {"opencl", [](const json &params) {
                       return std::make_unique<lang::okl::openclParser>(params);
                   }},
                  {"metal", [](const json &params) {
                       return std::make_unique<lang::okl::metalParser>(params);
                   }},
                  {"dpcpp", [](const json &params) {
                       return std::make_unique<lang::okl::dpcppParser>(params);
                   }}
              };

          const bool printLauncher = options["launcher"];
          const std::string filename = arguments[0];

          if (!io::exists(filename)) {
            printError("File [" + filename + "] doesn't exist" );
            ::exit(1);
          }

          auto parserIt = originalParserBackends.find(mode);
          if(parserIt == originalParserBackends.end()) {
            printError("Unable to translate for mode [" + originalMode + "]");
            ::exit(1);
          }

          std::unique_ptr<lang::parser_t> parser = parserIt->second(kernelProps);
          parser->parseFile(filename);

          bool success = parser->succeeded();
          if (!success) {
            ::exit(1);
          }

          if (options["verbose"]) {
            json translationInfo;
            // Filename
            translationInfo["translate_info/filename"] = io::expandFilename(filename);
            // Date information
            translationInfo["translate_info/date"] = sys::date();
            translationInfo["translate_info/human_date"] = sys::humanDate();
            // Version information
            translationInfo["translate_info/occa_version"] = OCCA_VERSION_STR;
            translationInfo["translate_info/okl_version"] = OKL_VERSION_STR;
            // Kernel properties
            translationInfo["kernel_properties"] = kernelProps;

            io::stdout
                << "/* Translation Info:\n"
                << translationInfo
                << "*/\n";
          }

          if (printLauncher && ((mode == "cuda")
                                || (mode == "hip")
                                || (mode == "opencl")
                                || (mode == "dpcpp")
                                || (mode == "metal"))) {
            lang::parser_t *launcherParser = &(((occa::lang::okl::withLauncher*) parser.get())->launcherParser);
            io::stdout << launcherParser->toString();
          } else {
            io::stdout << parser->toString();
          }
          return true;
        }
    }

#if BUILD_WITH_CLANG_BASED_TRANSPILER
    namespace v3 {
        bool runTranspiler(const json &options,
                           const json &arguments,
                           const json &kernelProps,
                           const std::string &originalMode,
                           const std::string &mode)
        {

          auto onFileNotExists = [](const std::string &file) {
              printError("File [" + file + "] doesn't exist" );
              ::exit(1);
          };

          auto onWrongBackend = [](const std::string &m) {
              printError("Unsupported target backend: [" + m + "]");
              ::exit(1);
          };

          auto onFail = [](const std::vector<oklt::Error> &errors) {
              std::stringstream ss;
              for(const auto &err: errors) {
                  ss << err.desc << std::endl;
              }
              printError(ss.str());
              ::exit(1);
          };

          std::string filename = arguments[0];
          auto onSuccess = [&](const oklt::UserOutput &output, bool hasLauncher) -> bool {
              const bool printLauncher = options["launcher"];

              if (options["verbose"]) {
                  json translationInfo;
                  // Filename
                  translationInfo["translate_info/filename"] = io::expandFilename(filename);
                  // Date information
                  translationInfo["translate_info/date"] = sys::date();
                  translationInfo["translate_info/human_date"] = sys::humanDate();
                  // Version information
                  translationInfo["translate_info/occa_version"] = OCCA_VERSION_STR;
                  translationInfo["translate_info/okl_version"] = OKL_VERSION_STR;
                  // Kernel properties
                  translationInfo["kernel_properties"] = kernelProps;

                  io::stdout
                      << "/* Translation Info:\n"
                      << translationInfo
                      << "*/\n";
              }

              if(printLauncher && hasLauncher) {
                  io::stdout << output.launcher.source;
              } else {
                  io::stdout << output.kernel.source;
              }

              return true;
          };

          transpiler::Transpiler transpiler(onSuccess, onFail, onFileNotExists, onWrongBackend);
          return transpiler.run(filename, mode, kernelProps);
        }
    }
#endif
    bool runTranspiler(const json &options,
                       const json &arguments,
                       const json &kernelProps,
                       const std::string &originalMode,
                       const std::string &mode)
    {
        int transpilerVersion = getTranspilerVersion(options);
    #ifdef BUILD_WITH_CLANG_BASED_TRANSPILER
        if(transpilerVersion > 2) {
            return v3::runTranspiler(options, arguments, kernelProps, originalMode, mode);
        }
    #endif
        if (transpilerVersion > 2) {
            printError("OCCA compiler is built without BUILD_WITH_CLANG_BASED_TRANSPILER support");
            return false;
        }
        return v2::runTranspiler(options, arguments, kernelProps, originalMode, mode);
    }


    bool runTranslate(const json &args) {
      const json &options = args["options"];
      const json &arguments = args["arguments"];

      const std::string originalMode = options["mode"];
      const std::string mode = lowercase(originalMode);

      json kernelProps = getOptionProperties(options["kernel-props"]);
      kernelProps["mode"] = mode;
      kernelProps["defines"].asObject() += getOptionDefines(options["define"]);
      kernelProps["okl/include_paths"] = options["include-path"];
      return runTranspiler(options, arguments, kernelProps, originalMode, mode);
    }

    bool runCompile(const json &args) {
      const json &options = args["options"];
      const json &arguments = args["arguments"];

      const std::string filename = arguments[0];
      const std::string kernelName = arguments[1];

      if (!io::exists(filename)) {
        printError("File [" + filename + "] doesn't exist" );
        ::exit(1);
      }

      json deviceProps = getOptionProperties(options["device-props"]);

      json kernelProps = getOptionProperties(options["kernel-props"]);
      kernelProps["verbose"] = kernelProps.get("verbose", true);
      kernelProps["okl/include_paths"] = options["include-path"];
      kernelProps["defines"].asObject() += getOptionDefines(options["define"]);
      kernelProps["transpiler-version"] = getTranspilerVersion(options);

      device device(deviceProps);
      device.buildKernel(filename, kernelName, kernelProps);

      return true;
    }

    bool runEnv(const json &args) {
      io::stdout << "  Basic:\n"
                 << "    - OCCA_DIR                   : " << envEcho("OCCA_DIR") << "\n"
                 << "    - OCCA_CACHE_DIR             : " << envEcho("OCCA_CACHE_DIR") << "\n"
                 << "    - OCCA_VERBOSE               : " << envEcho("OCCA_VERBOSE") << "\n"
                 << "    - OCCA_UNSAFE                : " << OCCA_UNSAFE << "\n"

                 << "  Makefile:\n"
                 << "    - CXX                        : " << envEcho("CXX") << "\n"
                 << "    - CXXFLAGS                   : " << envEcho("CXXFLAGS") << "\n"
                 << "    - CC                         : " << envEcho("CC") << "\n"
                 << "    - CFLAGS                     : " << envEcho("CFLAGS") << "\n"
                 << "    - FC                         : " << envEcho("FC") << "\n"
                 << "    - FCFLAGS                    : " << envEcho("FCFLAGS") << "\n"
                 << "    - LDFLAGS                    : " << envEcho("LDFLAGS") << "\n"

                 << "  Backend Support:\n"
                 << "    - OCCA_OPENMP_ENABLED        : " << envEcho("OCCA_OPENMP_ENABLED", OCCA_OPENMP_ENABLED) << "\n"
                 << "    - OCCA_CUDA_ENABLED          : " << envEcho("OCCA_CUDA_ENABLED", OCCA_CUDA_ENABLED) << "\n"
                 << "    - OCCA_HIP_ENABLED           : " << envEcho("OCCA_HIP_ENABLED", OCCA_HIP_ENABLED) << "\n"
                 << "    - OCCA_OPENCL_ENABLED        : " << envEcho("OCCA_OPENCL_ENABLED", OCCA_OPENCL_ENABLED) << "\n"
                 << "    - OCCA_DPCPP_ENABLED         : " << envEcho("OCCA_DPCPP_ENABLED", OCCA_DPCPP_ENABLED) << "\n"
                 << "    - OCCA_METAL_ENABLED         : " << envEcho("OCCA_METAL_ENABLED", OCCA_METAL_ENABLED) << "\n"

                 << "  Run-Time Options:\n"
                 << "    - OCCA_CXX                   : " << envEcho("OCCA_CXX") << "\n"
                 << "    - OCCA_CXXFLAGS              : " << envEcho("OCCA_CXXFLAGS") << "\n"
                 << "    - OCCA_CC                    : " << envEcho("OCCA_CC") << "\n"
                 << "    - OCCA_CFLAGS                : " << envEcho("OCCA_CFLAGS") << "\n"
                 << "    - OCCA_LDFLAGS               : " << envEcho("OCCA_LDFLAGS") << "\n"
                 << "    - OCCA_COMPILER_SHARED_FLAGS : " << envEcho("OCCA_COMPILER_SHARED_FLAGS") << "\n"
                 << "    - OCCA_INCLUDE_PATH          : " << envEcho("OCCA_INCLUDE_PATH") << "\n"
                 << "    - OCCA_LIBRARY_PATH          : " << envEcho("OCCA_LIBRARY_PATH") << "\n"
                 << "    - OCCA_KERNEL_PATH           : " << envEcho("OCCA_KERNEL_PATH") << "\n"
                 << "    - OCCA_OPENCL_COMPILER_FLAGS : " << envEcho("OCCA_OPENCL_COMPILER_FLAGS") << "\n"
                 << "    - OCCA_DPCPP_COMPILER        : " << envEcho("OCCA_DPCPP_COMPILER") << "\n"
                 << "    - OCCA_DPCPP_COMPILER_FLAGS  : " << envEcho("OCCA_DPCPP_COMPILER_FLAGS") << "\n"
                 << "    - OCCA_CUDA_COMPILER         : " << envEcho("OCCA_CUDA_COMPILER") << "\n"
                 << "    - OCCA_CUDA_COMPILER_FLAGS   : " << envEcho("OCCA_CUDA_COMPILER_FLAGS") << "\n"
                 << "    - OCCA_HIP_COMPILER          : " << envEcho("OCCA_HIP_COMPILER") << "\n"
                 << "    - OCCA_HIP_COMPILER_FLAGS    : " << envEcho("OCCA_HIP_COMPILER_FLAGS") << "\n";
      return true;
    }

    bool runInfo(const json &args) {
      occa::printModeInfo();
      return true;
    }

    bool runModes(const json &args) {
      for (auto &it : getModeMap()) {
        io::stdout << it.second->name() << '\n';
      }
      return true;
    }

    bool runBashAutocomplete(const json &args) {
      const json &arguments = args["arguments"];

      // Build the occa command again to run the command autocomplete
      cli::command occaCommand = occa::bin::buildOccaCommand();

      if (!arguments.size() || ((std::string) arguments[0] != "--")) {
        occaCommand.printBashAutocomplete("occa autocomplete bash");
      } else {
        // Remove the "--" argument
        strVector cmdArgs = arguments.toVector<std::string>();
        cmdArgs.erase(cmdArgs.begin());

        occaCommand.printBashSuggestions(cmdArgs);
      }

      return true;
    }

    cli::command buildOccaCommand() {
      cli::command versionCommand;
      versionCommand
          .withName("version")
          .withCallback(runVersion)
          .withDescription("Prints OCCA version")
          .addOption(cli::option("okl",
                                 "Print the OKL language version")
                     .stopsExpansion());

      cli::command clearCommand;
      clearCommand
          .withName("clear")
          .withCallback(runClear)
          .withDescription("Clears cached files and cache locks")
          .addOption(cli::option('a', "all",
                                 "Clear cached kernels and/or locks.")
                     .stopsExpansion())
          .addOption(cli::option("kernels",
                                 "Clear cached kernels."))
          .addOption(cli::option('l', "locks",
                                 "Clear cache locks"))
          .addOption(cli::option('y', "yes",
                                 "Automatically answer everything with [y/yes]"));

      cli::command translateCommand;
      translateCommand
          .withName("translate")
          .withCallback(runTranslate)
          .withDescription("Translate kernels")
          .addOption(cli::option('m', "mode",
                                 "Output mode (Default: Serial)")
                     .withArg()
                     .expandsFunction([&](const json &args) {
                         strVector suggestions;
                         for (auto &it : getModeMap()) {
                           suggestions.push_back(it.second->name());
                         }
                         return suggestions;
                       }))
          .addOption(cli::option('l', "launcher",
                                 "Output the launcher source instead"))
          .addOption(cli::option('k', "kernel-props",
                                 "Kernel properties")
                     .reusable()
                     .withArg())
          .addOption(cli::option('I', "include-path",
                                 "Add additional include path")
                     .reusable()
                     .withArg())
          .addOption(cli::option('D', "define",
                                 "Add additional define")
                     .reusable()
                     .withArg())
          .addOption(cli::option('v', "verbose",
                                 "Verbose output"))
          .addOption(cli::option('t', "transpiler-version",
                                 "provide transpiler version")
                     .reusable().withArg())
          .addArgument(cli::argument("FILE",
                                     "An .okl file")
                       .isRequired()
                       .expandsFiles());

      cli::command compileCommand;
      compileCommand
          .withName("compile")
          .withCallback(runCompile)
          .withDescription("Compile kernels")
          .addOption(cli::option('d', "device-props",
                                 "Device properties")
                     .reusable()
                     .withArg())
          .addOption(cli::option('k', "kernel-props",
                                 "Kernel properties")
                     .reusable()
                     .withArg())
          .addOption(cli::option('I', "include-path",
                                 "Add additional include path")
                     .reusable()
                     .withArg())
          .addOption(cli::option('D', "define",
                                 "Add additional define")
                     .reusable()
                     .withArg())
          .addOption(cli::option('t', "transpiler-version",
                                 "provide transpiler version")
                     .reusable().withArg())
          .addArgument(cli::argument("FILE",
                                     "An .okl file")
                       .isRequired()
                       .expandsFiles())
          .addArgument(cli::argument("KERNEL",
                                     "Kernel name")
                       .isRequired());

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

      return (
        cli::command()
        .withName("occa")
        .withDescription("Helpful utilities related to OCCA workflows")
        .requiresCommand()
        .addCommand(versionCommand)
        .addCommand(clearCommand)
        .addCommand(translateCommand)
        .addCommand(compileCommand)
        .addCommand(envCommand)
        .addCommand(infoCommand)
        .addCommand(modesCommand)
        .addCommand(autocompleteCommand)
      );
    }
  }
}
