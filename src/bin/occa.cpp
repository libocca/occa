#include <fstream>

#include <occa.hpp>
#include <occa/bin/occa.hpp>
#include <occa/lang/modes/serial.hpp>
#include <occa/lang/modes/openmp.hpp>
#include <occa/lang/modes/cuda.hpp>
#include <occa/lang/modes/hip.hpp>
#include <occa/lang/modes/opencl.hpp>
#include <occa/lang/modes/metal.hpp>

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

    properties getOptionProperties(const json &opt) {
      properties props;
      for (int i = 0; i < opt.size(); ++i) {
        props += properties((std::string) opt[i]);
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

    bool runTranslate(const json &args) {
      const json &options = args["options"];
      const json &arguments = args["arguments"];

      const std::string originalMode = options["mode"];
      const std::string mode = lowercase(originalMode);

      const bool printLauncher = options["launcher"];
      const std::string filename = arguments[0];

      if (!io::exists(filename)) {
        printError("File [" + filename + "] doesn't exist" );
        ::exit(1);
      }

      properties kernelProps = getOptionProperties(options["kernel-props"]);
      kernelProps["mode"] = mode;
      kernelProps["defines"].asObject() += getOptionDefines(options["define"]);
      kernelProps["okl/include_paths"] = options["include-path"];

      lang::parser_t *parser = NULL;
      lang::parser_t *launcherParser = NULL;
      if (mode == "" || mode == "serial") {
        parser = new lang::okl::serialParser(kernelProps);
      } else if (mode == "openmp") {
        parser = new lang::okl::openmpParser(kernelProps);
      } else if (mode == "cuda") {
        parser = new lang::okl::cudaParser(kernelProps);
      } else if (mode == "hip") {
        parser = new lang::okl::hipParser(kernelProps);
      } else if (mode == "opencl") {
        parser = new lang::okl::openclParser(kernelProps);
      } else if (mode == "metal") {
        parser = new lang::okl::metalParser(kernelProps);
      }

      if (!parser) {
        printError("Unable to translate for mode [" + originalMode + "]");
        ::exit(1);
      }

      parser->parseFile(filename);

      bool success = parser->succeeded();
      if (!success) {
        delete parser;
        ::exit(1);
      }

      if (options["verbose"]) {
        properties translationInfo;
        // Filename
        translationInfo["translate_info/filename"] = io::filename(filename);
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
                            || (mode == "metal"))) {
        launcherParser = &(((occa::lang::okl::withLauncher*) parser)->launcherParser);
        io::stdout << launcherParser->toString();
      } else {
        io::stdout << parser->toString();
      }

      delete parser;
      if (!success) {
        ::exit(1);
      }
      return true;
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

      properties deviceProps = getOptionProperties(options["device-props"]);

      properties kernelProps = getOptionProperties(options["kernel-props"]);
      kernelProps["verbose"] = kernelProps.get("verbose", true);
      kernelProps["okl/include_paths"] = options["include-path"];
      kernelProps["defines"].asObject() += getOptionDefines(options["define"]);

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
                 << "    - FC                         : " << envEcho("FC") << "\n"
                 << "    - FCFLAGS                    : " << envEcho("FCFLAGS") << "\n"
                 << "    - LDFLAGS                    : " << envEcho("LDFLAGS") << "\n"

                 << "  Backend Support:\n"
                 << "    - OCCA_OPENMP_ENABLED        : " << envEcho("OCCA_OPENMP_ENABLED", OCCA_OPENMP_ENABLED) << "\n"
                 << "    - OCCA_CUDA_ENABLED          : " << envEcho("OCCA_CUDA_ENABLED", OCCA_CUDA_ENABLED) << "\n"
                 << "    - OCCA_HIP_ENABLED           : " << envEcho("OCCA_HIP_ENABLED", OCCA_HIP_ENABLED) << "\n"
                 << "    - OCCA_OPENCL_ENABLED        : " << envEcho("OCCA_OPENCL_ENABLED", OCCA_OPENCL_ENABLED) << "\n"
                 << "    - OCCA_METAL_ENABLED         : " << envEcho("OCCA_METAL_ENABLED", OCCA_METAL_ENABLED) << "\n"

                 << "  Run-Time Options:\n"
                 << "    - OCCA_CXX                   : " << envEcho("OCCA_CXX") << "\n"
                 << "    - OCCA_CXXFLAGS              : " << envEcho("OCCA_CXXFLAGS") << "\n"
                 << "    - OCCA_LDFLAGS               : " << envEcho("OCCA_LDFLAGS") << "\n"
                 << "    - OCCA_COMPILER_SHARED_FLAGS : " << envEcho("OCCA_COMPILER_SHARED_FLAGS") << "\n"
                 << "    - OCCA_INCLUDE_PATH          : " << envEcho("OCCA_INCLUDE_PATH") << "\n"
                 << "    - OCCA_LIBRARY_PATH          : " << envEcho("OCCA_LIBRARY_PATH") << "\n"
                 << "    - OCCA_KERNEL_PATH           : " << envEcho("OCCA_KERNEL_PATH") << "\n"
                 << "    - OCCA_OPENCL_COMPILER_FLAGS : " << envEcho("OCCA_OPENCL_COMPILER_FLAGS") << "\n"
                 << "    - OCCA_CUDA_COMPILER         : " << envEcho("OCCA_CUDA_COMPILER") << "\n"
                 << "    - OCCA_CUDA_COMPILER_FLAGS   : " << envEcho("OCCA_CUDA_COMPILER_FLAGS") << "\n"
                 << "    - OCCA_HIP_COMPILER          : " << envEcho("OCCA_HIP_COMPILER") << "\n"
                 << "    - OCCA_HIP_COMPILER_FLAGS    : " << envEcho("OCCA_HIP_COMPILER_FLAGS") << "\n";
      return true;
    }

    bool runInfo(const json &args) {
      printModeInfo();
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
        strVector cmdArgs = arguments.getArray<std::string>();
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