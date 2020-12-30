#include <algorithm>
#include <iomanip>
#include <map>
#include <sstream>

#include <occa/internal/utils/cli.hpp>
#include <occa/internal/utils/env.hpp>
#include <occa/internal/io.hpp>
#include <occa/internal/utils/lex.hpp>
#include <occa/internal/utils/string.hpp>

namespace occa {
  namespace cli {
    namespace pretty {
      void printDescription(std::ostream &out,
                            const std::string &description) {
        printDescription(out,
                         0, MAX_NAME_COLUMN_WIDTH + MAX_DESC_COLUMN_WIDTH,
                         description);
      }

      void printDescription(std::ostream &out,
                            const int indent, const int width,
                            const std::string &description) {
        std::stringstream ss;

        // Print the description across multiple lines if needed
        const char *desc_c = &(description[0]);
        while (*desc_c) {
          const char *start = desc_c;
          lex::skipToWhitespace(desc_c);
          const std::string word(start, desc_c - start);

          if ((int) (ss.str().size() + word.size()) >= width) {
            out << ss.str()
                << '\n' << std::string(indent, ' ');
            ss.str("");
          }
          ss << word;

          start = desc_c;
          lex::skipWhitespace(desc_c);
          const std::string space(start, desc_c - start);

          if ((int) (ss.str().size() + space.size()) >= width) {
            ss << std::string(width - ss.str().size(), ' ');
          } else {
            ss << space;
          }
        }
        if (ss.str().size()) {
          out << ss.str();
          ss.str("");
        }
        out << '\n';
      }
    }

    //---[ Printable ]------------------
    printable::~printable() {}
    //==================================

    //---[ Option ]---------------------
    option::option() :
      shortname('\0'),
      flags(0),
      requiredArgs(0) {

      name="";
      description="";
    }

    option::option(const char shortname_,
                   const std::string &name_,
                   const std::string &description_) :
      shortname(shortname_),
      flags(0),
      requiredArgs(0) {

      name = name_;
      description = description_;
    }

    option::option(const std::string &name_,
                   const std::string &description_) :
      shortname('\0'),
      flags(0),
      requiredArgs(0) {

      name = name_;
      description = description_;
    }

    option::~option() {}

    option option::isRequired(const bool required) {
      if (!required) {
        return *this;
      }
      option opt = *this;
      opt.flags |= flags_t::isRequired;
      return opt;
    }

    option option::reusable() {
      option opt = *this;
      opt.flags |= flags_t::reusable;
      return opt;
    }

    option option::withDefaultValue(const json &defaultValue_) {
      option opt = *this;
      opt.defaultValue = defaultValue_;
      return opt;
    }

    option option::withArg() {
      option opt = *this;
      opt.requiredArgs = 1;
      return opt;
    }

    option option::withArgs(const int requiredArgs_) {
      option opt = *this;
      opt.requiredArgs = requiredArgs_;
      return opt;
    }

    option option::stopsExpansion() {
      option opt = *this;
      opt.flags |= flags_t::stopsExpansion;
      return opt;
    }

    option option::expandsFiles() {
      option opt = *this;
      opt.flags |= flags_t::expandsFiles;
      return opt;
    }

    option option::expandsFunction(functionExpansionCallback expansionFunction_) {
      option opt = *this;
      opt.flags |= flags_t::expandsFunction;
      opt.expansionFunction = expansionFunction_;
      return opt;
    }

    std::string option::getShortnameFlag() const {
      if (!shortname) {
        return "";
      }
      std::string flag = "-";
      flag += shortname;
      return flag;
    }

    std::string option::getNameFlag() const {
      if (name.size()) {
        return "--" + name;
      }
      return "";
    }

    bool option::getIsRequired() {
      return (flags & flags_t::isRequired);
    }

    bool option::getReusable() {
      return (flags & flags_t::reusable);
    }

    bool option::getStopsExpansion() {
      return (flags & flags_t::stopsExpansion);
    }

    bool option::getExpandsFiles() {
      return (flags & flags_t::expandsFiles);
    }

    bool option::getExpandsFunction() {
      return (flags & flags_t::expandsFunction);
    }

    bool option::hasDefaultValue() {
      return defaultValue.isInitialized();
    }

    std::string option::getPrintName() const {
      std::string ret;
      if (shortname) {
        ret += getShortnameFlag();
        if (name.size()) {
          ret += ", ";
          ret += getNameFlag();
        }
      } else {
        ret += "    ";
        ret += getNameFlag();
      }
      return ret;
    }

    std::string option::toString() const {
      return strip(getPrintName());
    }

    bool operator < (const option &l, const option &r) {
      const char leftSN = l.shortname ? l.shortname : l.name[0];
      const char rightSN = r.shortname ? r.shortname : r.name[0];
      if (leftSN != rightSN) {
        return leftSN < rightSN;
      }
      if (l.shortname || r.shortname) {
        return l.shortname;
      }
      return l.name < r.name;
    }

    std::ostream& operator << (std::ostream &out, const option &opt) {
      out << opt.toString();
      return out;
    }
    //==================================

    //---[ Argument ]-------------------
    argument::argument() :
      option() {}

    argument::argument(const option &opt) :
      option(opt) {}

    argument::argument(const std::string &name_,
                       const std::string &description_) :
      option(name_, description_) {}

    argument::~argument() {}

    std::string argument::getPrintName() const {
      return name;
    }

    std::string argument::toString() const {
      return name;
    }
    //==================================

    //---[ Parser ]---------------------
    parser::parser() {}

    parser::~parser() {}

    std::string parser::getPrintName() const {
      return name;
    }

    bool parser::isLongOption(const std::string &arg) {
      return ((arg.size() > 2) &&
              (arg[0] == '-') &&
              (arg[1] == '-') &&
              (arg[2] != '-'));
    }

    bool parser::isShortOption(const std::string &arg) {
      return ((arg.size() == 2) &&
              (arg[0] == '-') &&
              (arg[1] != '-'));
    }

    bool parser::hasShortOption(const std::string &arg) {
      return ((arg.size() >= 2) &&
              (arg[0] == '-') &&
              (arg[1] != '-'));
    }

    bool parser::isOption(const std::string &arg) {
      return (isLongOption(arg) ||
              isShortOption(arg));
    }

    option* parser::getShortOption(const char opt,
                                   const bool errorIfMissing) {
      const int optCount = (int) options.size();
      for (int i = 0; i < optCount; ++i) {
        if (options[i].shortname == opt) {
          return &(options[i]);
        }
      }

      if (errorIfMissing) {
        std::stringstream ss;
        ss << "Unknown option [-" << opt << ']';
        fatalError(ss.str());
      }

      return NULL;
    }

    option* parser::getLongOption(const std::string &opt,
                                  const bool errorIfMissing) {
      const int optCount = (int) options.size();
      for (int i = 0; i < optCount; ++i) {
        if (options[i].name == opt) {
          return &(options[i]);
        }
      }

      if (errorIfMissing) {
        std::stringstream ss;
        ss << "Unknown option [--" << opt << ']';
        fatalError(ss.str());
      }

      return NULL;
    }

    option* parser::getOption(const std::string &arg,
                              const bool errorIfMissing) {
      if (isLongOption(arg)) {
        return getLongOption(arg.substr(2), errorIfMissing);
      }
      return (isShortOption(arg)
              ? getShortOption(arg[1], errorIfMissing)
              : NULL);
    }

    bool parser::hasOptionalArg() {
      const int argumentCount = (int) arguments.size();
      return (argumentCount &&
              !arguments[argumentCount - 1].getIsRequired());
    }

    parser& parser::withDescription(const std::string &description_) {
      description = description_;
      return *this;
    }

    parser& parser::addArgument(const argument &arg) {
      OCCA_ERROR("Cannot add " << arguments[arguments.size() - 1]
                 << ", an optional argument has already been added\n",
                 !hasOptionalArg());

      arguments.push_back(arg);

      return *this;
    }

    parser& parser::addOption(const option &option) {
      options.push_back(option);
      return *this;
    }

    strVector parser::vectorizeArgs(const int argc, const char **argv) {
      strVector args;
      for (int i = 0; i < argc; ++i) {
        args.push_back(argv[i]);
      }
      return args;
    }

    // Create list of args, splitting combined short options
    // -abc could split as
    //   -> -a -b -c
    //   -> -a bc
    strVector parser::splitShortOptionArgs(const strVector &args) {
      const int argc = (int) args.size();
      bool splitShortOptions = true;

      strVector newArgs;
      newArgs.reserve(argc);

      for (int i = 0; i < argc; ++i) {
        const std::string arg = args[i];
        const int argSize = (int) arg.size();

        // Check for short option
        if (!splitShortOptions ||
            !hasShortOption(arg) ||
            !getShortOption(arg[1], false)) {
          splitShortOptions = (arg != "--");
          newArgs.push_back(arg);
          continue;
        }

        // Split short options
        for (int ci = 1; ci < argSize; ++ci) {
          option &opt = *(getShortOption(arg[ci]));

          std::string shortArg = "-";
          shortArg += opt.shortname;
          newArgs.push_back(shortArg);

          if ((ci < (argSize - 1)) &&
              (opt.requiredArgs > 0)) {
            newArgs.push_back(arg.substr(ci + 1));
            break;
          }
        }
      }

      return newArgs;
    }

    occa::json parser::parseArgs(const int argc, const char **argv) {
      return parseArgs(vectorizeArgs(argc, argv));
    }

    occa::json parser::parseArgs(const strVector &args_,
                                 const bool supressErrors) {
      strVector args = splitShortOptionArgs(args_);
      const int argc = (int) args.size();

      // Set name to script name
      name = args[0];

      bool hasHelpOption = hasCustomHelpOption();
      if (!hasHelpOption) {
        addHelpOption();
      }

      occa::json parsedArgs(json::object_);
      occa::json &jOptions      = parsedArgs["options"].asObject();
      occa::json &jOptionsOrder = parsedArgs["options_order"].asArray();
      occa::json &jArguments    = parsedArgs["arguments"].asArray();
      setOptionDefaults(jOptions);

      // Make a list of used options to check required options later
      std::map<std::string, bool> usedOptions;

      bool checkOptions = true;
      for (int i = 1; i < argc; ++i) {
        const std::string &arg = args[i];

        // Stop parsing args after encountering `--`
        if (arg == "--") {
          for (int i2 = i; i2 < argc; ++i2) {
            jArguments += args[i2];
          }
          break;
        }

        option *opt = NULL;
        if (checkOptions) {
          opt = getOption(arg);
        }

        // No option
        if (!opt) {
          checkOptions = (arg == "==");
          jArguments += arg;
          continue;
        }

        if ((opt->name == "help") &&
            !hasHelpOption) {
          if (supressErrors) {
            return parsedArgs;
          }
          printUsage(name);
          ::exit(0);
        }

        occa::json &jOpt = jOptions[opt->name];
        jOptionsOrder += opt->name;
        usedOptions[opt->name] = true;

        // True/False option
        if (opt->requiredArgs <= 0) {
          jOpt = true;
          continue;
        }

        // Add argument to current option
        for (int ai = 0; ai < opt->requiredArgs; ++ai) {
          ++i;

          option *subOpt = NULL;
          if (i < argc) {
            subOpt = getOption(args[i]);
          }
          if (subOpt || (i >= argc)) {
            if (supressErrors) {
              return parsedArgs;
            }
            std::stringstream ss;
            ss << "Incorrect arguments for [" << arg << ']';
            fatalError(ss.str());
          }

          // Check if we need to store value or entry in array
          if ((opt->requiredArgs > 1) ||
              opt->getReusable()) {
            jOpt += args[i];
          } else {
            jOpt = args[i];
          }
        }
      }

      // Check required options
      const int optCount = (int) options.size();
      for (int i = 0; i < optCount; ++i) {
        option &opt = options[i];
        if (!opt.getIsRequired()) {
          continue;
        }
        if (usedOptions.find(opt.name) == usedOptions.end()) {
          if (supressErrors) {
            return parsedArgs;
          }
          std::stringstream ss;
          ss << "Missing required option [" << opt.toString() << ']';
          fatalError(ss.str());
        }
      }

      // Check required arguments
      const int argCount = (int) jArguments.array().size();
      const int reqArgCount = (int) arguments.size() - hasOptionalArg();
      if (argCount < reqArgCount) {
        if (supressErrors) {
          return parsedArgs;
        }
        if (argCount == 0) {
          printUsage(name);
          ::exit(0);
        }
        fatalError("Incorrect number of arguments");
      }

      return parsedArgs;
    }

    bool parser::hasCustomHelpOption() {
      const int optCount = (int) options.size();
      for (int i = 0; i < optCount; ++i) {
        option &opt = options[i];
        if (opt.name == "help") {
          return true;
        }
      }
      return false;
    }

    void parser::addHelpOption() {
      bool hasShortOption = false;
      const int optCount = (int) options.size();
      for (int i = 0; i < optCount; ++i) {
        option &opt = options[i];
        if (opt.name == "help") {
          return;
        }
        hasShortOption = (opt.shortname == 'h');
      }
      options.push_back(
        option(hasShortOption ? '\0' : 'h',
               "help",
               "Print usage")
      );
    }

    void parser::setOptionDefaults(occa::json &jOptions) {
      const int optCount = (int) options.size();
      for (int i = 0; i < optCount; ++i) {
        option &opt = options[i];
        occa::json &jOpt = jOptions[opt.name];

        if (opt.hasDefaultValue()) {
          jOpt = opt.defaultValue;
        }
        else if (opt.requiredArgs <= 0) {
          jOpt = false;
        }
        else if ((opt.requiredArgs > 1) ||
                 opt.getReusable()) {
          jOpt.asArray();
        }
        else {
          jOpt = "";
        }
      }
    }

    void parser::printUsage(const std::string &program,
                            std::ostream &out) {
      out << "\nUsage: " << program;

      if (options.size()) {
        out << " [OPTIONS]";
      }

      const int argumentCount = (int) arguments.size();
      for (int i = 0; i < argumentCount; ++i) {
        option &argument = arguments[i];
        const bool repeats = hasRepetitiveArg && (i == (argumentCount - 1));
        if (argument.getIsRequired()) {
          out << ' ' << argument.name;
        } else if (!repeats) {
          out << " [" << argument.name << ']';
        }

        if (repeats) {
          out << " [" << argument.name << "...]";
        }
      }
      out << "\n\n";
      if (description.size()) {
        pretty::printDescription(out, description);
      } else {
        out << '\n';
      }
      out << '\n';

      printRequired(out);

      std::sort(options.begin(), options.end());
      pretty::printEntries("Arguments", arguments, out);
      pretty::printEntries("Options", options, out);
    }

    void parser::printRequired(std::ostream &out) {}

    void parser::fatalError(const std::string &message) {
      std::cerr << red("Error") << ": " << message << '\n';
      printUsage(name, std::cerr);
      ::exit(1);
    }
    //==================================

    //---[ Command ]--------------------
    command::command() :
      commandIsRequired(false),
      callback(NULL) {}

    command::~command() {}

    command& command::withName(const std::string &name_) {
      name = name_;
      return *this;
    }

    command& command::withDescription(const std::string &description_) {
      description = description_;
      return *this;
    }

    command& command::withCallback(commandCallback callback_) {
      callback = callback_;
      return *this;
    }

    command& command::withFunctionExpansion(std::string expansion) {
      expansionFunction = expansion;
      return *this;
    }

    command* command::getCommand(const std::string &name_) {
      for (auto &comm : commands) {
        if (comm.name == name_) {
          return &comm;
        }
      }
      return NULL;
    }

    void command::fillProgram(std::string &program) {
      program += commandPath;
      if (name.size()) {
        if (commandPath.size()) {
          program += ' ';
        }
        program += name;
      }
    }

    void command::printUsage(std::ostream &out) {
      printUsage("", out);
    }

    void command::printUsage(const std::string &program,
                             std::ostream &out) {
      std::string newProgram;
      fillProgram(newProgram);

      parser::printUsage(newProgram, out);
    }

    void command::printRequired(std::ostream &out) {
      std::sort(commands.begin(), commands.end());
      pretty::printEntries("Commands", commands, out);
    }

    command& command::requiresCommand() {
      commandIsRequired = true;
      return *this;
    }

    command& command::addCommand(const occa::cli::command &command_) {
      bool hasCommandArgument = false;
      for (auto &arg : arguments) {
        if (arg.name == "COMMAND") {
          hasCommandArgument = true;
          break;
        }
      }

      if (!hasCommandArgument) {
        addArgument(argument("COMMAND",
                             "Command to run")
                    .isRequired(commandIsRequired));
      }

      command_.setCommandPath(
        commandPath.size()
        ? commandPath + " " + name
        : name
      );

      commands.push_back(command_);

      return *this;
    }

    void command::setCommandPath(const std::string &commandPath_) const {
      if (!commandPath_.size()) {
        return;
      }

      commandPath = commandPath_;

      const std::string childCommandPath = commandPath + " " + name;
      for (auto &childCommand : commands) {
        childCommand.setCommandPath(childCommandPath);
      }
    }

    void command::run(const int argc, const char **argv) {
      run(vectorizeArgs(argc, argv));
    }

    void command::run(const strVector &args) {
      command *lastCommand = NULL;
      std::string lastCommandName;
      json lastCommandArgs;

      const bool successful = (
        findCommandAndArguments(args,
                                lastCommand,
                                lastCommandName,
                                lastCommandArgs)
      );

      if (!successful && commandIsRequired) {
        std::cerr << red("Error") << ": Unknown command [" << lastCommandName << "]\n";
        lastCommand->printUsage(std::cerr);
        ::exit(1);
      }

      if (lastCommand->callback) {
        if (lastCommand->callback(lastCommandArgs)) {
          return;
        }
        // Callback failed
        printUsage(std::cerr);
        ::exit(1);
      }
    }

    bool command::findCommandAndArguments(const strVector &shellArgs,
                                          command *&lastCommand,
                                          std::string &lastCommandName,
                                          json &lastCommandArgs,
                                          const bool supressErrors) {
      lastCommand = this;

      if (!shellArgs.size()) {
        lastCommandName = name;
        return true;
      }

      const bool hasCommands = commands.size();

      json parsedArgs = parseArgs(shellArgs, supressErrors);
      lastCommandArgs = parsedArgs;

      json &jArguments = parsedArgs["arguments"];
      strVector inputArgs = jArguments.toVector<std::string>();

      const int commandArg = arguments.size() - 1;
      command *comm = NULL;

      // Modify arguments and find command
      if (hasCommands &&
          inputArgs.size() &&
          commandArg < (int) inputArgs.size()) {

        // Remove command arguments
        jsonArray &jArgArray = jArguments.array();
        jArgArray = jsonArray(jArgArray.begin(),
                              jArgArray.begin() + commandArg + 1);

        // Extract command arguments
        inputArgs = strVector(inputArgs.begin() + commandArg,
                              inputArgs.end());

        lastCommandName = inputArgs[0];
        comm = getCommand(lastCommandName);
      }

      if (callback) {
        return true;
      }

      if (comm) {
        return comm->findCommandAndArguments(inputArgs,
                                             lastCommand,
                                             lastCommandName,
                                             lastCommandArgs,
                                             supressErrors);
      }

      return false;
    }

    void command::printBashAutocomplete(const std::string &fullBashCommand) {
      const std::string autocompleteName = "_occa_bash_autocomplete_" + name;

      std::cout << autocompleteName << "() {"                        << std::endl
                << "  local suggestions=$(" << fullBashCommand << " -- \"${COMP_WORDS[@]}\")" << std::endl
                << "  case \"${suggestions}\" in"                    << std::endl
                << "    " << BASH_STOPS_EXPANSION << ")"             << std::endl
                << "      compopt -o default -o nospace"             << std::endl
                << "      COMPREPLY=''"                              << std::endl
                << "      ;;"                                        << std::endl
                << "    " << BASH_EXPANDS_FILES << ")"               << std::endl
                << "      # Rely on the default options"             << std::endl
                << "      ;;"                                        << std::endl
                << "    *)"                                          << std::endl
                << "      COMPREPLY=($(compgen -W \"${suggestions}\" -- \"${COMP_WORDS[COMP_CWORD]}\"))" << std::endl
                << "      ;;"                                        << std::endl
                << "  esac"                                          << std::endl
                << "}"                                               << std::endl
                << ""                                                << std::endl
                << "complete -o bashdefault -o default -F " << autocompleteName << " " << name << std::endl;
    }

    void command::printBashSuggestions(const strVector &shellArgs) {
      // Examples:
      //    occa translate -[TAB]
      // -> [occa] [translate] [-]
      //
      //    occa translate [TAB]
      // -> [occa] [translate] []
      //
      strVector fullySetArgs = shellArgs;

      std::string autocompleteArg;
      if (fullySetArgs.size()) {
        // Remove the last argument we are trying to autocomplete
        // Note: It could be empty in the case the user is trying to
        //       find the next command/argument/option
        autocompleteArg = fullySetArgs.back();
        fullySetArgs.pop_back();
      }

      command *lastCommand = NULL;
      std::string lastCommandName;
      json lastCommandArgs;

      const bool supressErrors = true;
      findCommandAndArguments(fullySetArgs,
                              lastCommand,
                              lastCommandName,
                              lastCommandArgs,
                              supressErrors);

      strVector suggestions = (
        lastCommand->getCommandBashSuggestions(
          fullySetArgs,
          lastCommandArgs,
          autocompleteArg
        )
      );

      std::sort(suggestions.begin(), suggestions.end());

      for (auto &suggestion : suggestions) {
        io::stdout << suggestion << '\n';
      }
    }

    strVector command::getCommandBashSuggestions(const strVector &shellArgs,
                                                 const json &args,
                                                 const std::string &autocompleteArg) {
      /*
       * Print commands (getCommandSuggestions)
       *    [occa] []
       *    [occa] [tr]
       *
       * Print option flags (getOptionFlagSuggestions)
       *    [occa] [-]
       *    [occa] [--]
       *    [occa] [--help]
       *
       * Print option suggestions (getOptionSuggestions)
       *    [occa] [translate] [--mode] []
       *    [occa] [translate] [--mode] [CU]
       *    [occa] [translate] [--mode] [CUDA]
       */

      const int inputArguments = (int) args["arguments"].size();
      const json &shellOptions = args["options"];
      const strVector shellOptionsOrder = args["options_order"].toVector<std::string>();

      // Suggest an option since it starts with '-'
      if (startsWith(autocompleteArg, "-")) {
        return getOptionFlagSuggestions(shellOptionsOrder);
      }

      // Commands are only expanded when no options or arguments
      // have been passed
      if (!inputArguments && !shellOptionsOrder.size() && commands.size()) {
        return getCommandSuggestions();
      }

      const std::string lastShellArg = (
        shellArgs.size()
        ? shellArgs.back()
        : ""
      );

      // Check if we need to expand an option
      if (startsWith(lastShellArg, "-")) {
        option *lastOption = getOption(lastShellArg, false);
        if (lastOption) {
          return getOptionSuggestions(*lastOption,
                                      shellOptions[lastOption->name]);
        }
      }

      // Check if we need to suggest a command
      if (!inputArguments && !shellOptionsOrder.size() && commands.size()) {
        return getCommandSuggestions();
      }

      // Check if we need to autocomplete an argument
      if (inputArguments) {
        argument arg;

        if (inputArguments >= (int) arguments.size()) {
          // If we pass the argument count, check if we have a repetitive argument
          if (!hasRepetitiveArg) {
            return stopBashAutocomplete();
          }
          arg = arguments.back();
        } else {
          // Get the indexed argument
          arg = arguments[inputArguments - 1];
        }

        return getOptionSuggestions(arg);
      }

      // Autocomplete arguments
      if (arguments.size()) {
        return getOptionSuggestions(arguments[0]);
      }

      return stopBashAutocomplete();
    }

    strVector command::stopBashAutocomplete() {
      return {BASH_STOPS_EXPANSION};
    }

    strVector command::getBashFileExpansion() {
      return {"expands-files"};
    }

    strVector command::getCommandSuggestions() {
      strVector suggestions;

      for (auto &comm : commands) {
        suggestions.push_back(comm.name);
      }

      return suggestions;
    }

    strVector command::getOptionFlagSuggestions(const strVector &usedOptions) {
      strVector suggestions;

      // Filter out used options unless they are reusable
      for (auto &opt : options) {
        const bool addSuggestion = (
          opt.getReusable()
          || usedOptions.end() == (
            std::find(usedOptions.begin(), usedOptions.end(),
                      opt.name)
          )
        );
        if (!addSuggestion) {
          continue;
        }

        if (opt.shortname) {
          suggestions.push_back(opt.getShortnameFlag());
        }
        suggestions.push_back(opt.getNameFlag());
      }

      return suggestions;
    }

    strVector command::getOptionSuggestions(option &opt,
                                            const json &optArgs) {
      if(opt.getExpandsFiles()) {
        return getBashFileExpansion();
      }
      else if(opt.getExpandsFunction()) {
        return opt.expansionFunction(optArgs);
      }
      return strVector();
    }

    bool command::operator < (const command &comm) const {
      return name < comm.name;
    }
    //==================================

    //---[ JSON ]-----------------------
    json parse(const int argc,
               const char **argv,
               const char *config) {

      occa::cli::parser parser;
      json j = json::parse(config);

      if (j.has("description")) {
        parser.withDescription(j["description"]);
      }

      json options = j["options"].asArray();
      const int optionCount = options.size();
      for (int i = 0; i < optionCount; ++i) {
        json option_i = options[i];

        const std::string name = option_i.get<std::string>("name", "");
        const char shortname = option_i.get<std::string>("shortname", "\0")[0];
        const std::string description = option_i.get<std::string>("description", "");
        json defaultValue = option_i["default_value"];

        option opt(shortname, name, description);
        if (option_i.get("is_required", false)) {
          opt = opt.isRequired();
        }
        if (option_i.get("reusable", false)) {
          opt = opt.reusable();
        }
        if (option_i.get("with_arg", false)) {
          opt = opt.withArg();
        }
        if (option_i.has("with_args")) {
          opt = opt.withArgs(option_i.get("with_args", 0));
        }
        if (option_i.get("stops_expansion", false)) {
          opt = opt.stopsExpansion();
        }
        if (option_i.get("expands_files", false)) {
          opt = opt.expandsFiles();
        }
        if (defaultValue.isInitialized()) {
          opt = opt.withDefaultValue(defaultValue);
        }

        parser.addOption(opt);
      }

      return parser.parseArgs(argc, argv);
    }
    //==================================
  }
}
