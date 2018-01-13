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

#include <algorithm>
#include <iomanip>
#include <sstream>

#include "occa/tools/cli.hpp"
#include "occa/tools/env.hpp"
#include "occa/tools/io.hpp"
#include "occa/tools/lex.hpp"
#include "occa/tools/string.hpp"

namespace occa {
  namespace cli {
    //---[ Printable ]------------------
    printable::printable() {}

    std::string printable::getName() const {
      return name;
    }

    void printable::printDescription(std::ostream &out,
                                     const int indent, const int width,
                                     const std::string &description_) {
      std::stringstream ss;

      // Print the description across multiple lines if needed
      const char *desc_c = &(description_[0]);
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

    //---[ Option ]---------------------
    option::option() :
      shortname('\0'),
      flags(0),
      requiredArgs(0),
      expansionFunction("") {

      name="";
      description="";
    }

    option::option(const char shortname_,
                   const std::string &name_,
                   const std::string &description_) :
      shortname(shortname_),
      flags(0),
      requiredArgs(0),
      expansionFunction("") {

      name = name_;
      description = description_;
    }

    option option::isRequired() {
      option opt = *this;
      opt.flags |= flags_t::isRequired;
      return opt;
    }

    option option::reusable() {
      option opt = *this;
      opt.flags |= flags_t::reusable;
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

    option option::expandsFunction(const std::string &function) {
      option opt = *this;
      opt.flags |= flags_t::expandsFunction;
      opt.expansionFunction = function;
      return opt;
    }

    bool option::getIsRequired() {
      return (flags & flags_t::isRequired);
    }

    std::string option::getName() const {
      std::string ret;
      if (shortname) {
        ret += '-';
        ret += shortname;
        if (name.size()) {
          ret += ", --";
          ret += name;
        }
      } else {
        ret += "    --";
        ret += name;
      }
      return ret;
    }

    void option::printBashAutocomplete(const std::string &funcPrefix) {
      const std::string shortFlag = (!shortname   ? "" : ("-" + std::string(1, shortname)));
      const std::string longFlag  = (!name.size() ? "" : ("--" + name));
      const std::string shortFunc = funcPrefix + "_" + shortFlag;
      const std::string longFunc  = funcPrefix + "_" + longFlag;

      // Arguments for init_flag
      std::string flagArgs = "";
      if (shortname) {
        flagArgs += " \"" + shortFlag + "\"";
      }
      if (name.size()) {
        flagArgs += " \"" + longFlag + "\"";
      }

      // If the short and long flags exist, the short will call the long
      const std::string definitionFunc = longFlag.size() ? longFunc : shortFunc;

      // Methods to find the pair's flag
      if (shortFlag.size()) {
        std::cout << '\n'
                  << shortFunc << "-long() {\n"
                  << "    echo \"" << longFlag << "\"\n"
                  << "}\n";
      }
      if (longFlag.size()) {
        std::cout << '\n'
                  << longFunc << "-short() {\n"
                  << "    echo \"" << shortFlag << "\"\n"
                  << "}\n";
      }

      if (shortFlag.size() && longFlag.size()) {
        std::cout << '\n'
                  << shortFunc << "() {\n"
                  << "    " << longFunc << "\n"
                  << "}\n"
                  << "\n";
      }
      // The actual flag defines
      std::cout << '\n'
                << definitionFunc << "() {\n";

      if (flags & flags_t::stopsExpansion) {
        std::cout << "    compIsDone=true\n"
                  << "}\n";
        return;
      }

      std::cout << "    options=()\n"
                << "    flags=()\n";

      if (flags & flags_t::expandsFiles) {
        std::cout << "    expansions=(file)\n";
      } else if (flags & flags_t::expandsFunction) {
        std::cout << "    expansions=(func)\n"
                  << "    expansionFunction=\"" << expansionFunction << "\"\n";
      } else {
        std::cout << "    expansions=(same)\n";
      }

      std::cout << "    __occa_init_flag   " << flagArgs << "\n";
      if (flags & flags_t::reusable) {
        std::cout << "    __occa_reuse_flags " << flagArgs << "\n";
      }
      std::cout << "    __occa_compgen\n"
                << "}\n"
                << "\n";
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
      if (opt.shortname) {
        out << '-' << opt.shortname << '/';
      }
      out << "--" << opt.name;
      return out;
    }

    longOption::longOption() {}

    longOption::longOption(const option &opt) {
      shortname = opt.shortname;
      name = opt.name;
      description = opt.description;

      flags = opt.flags;
      expansionFunction = expansionFunction;
    }

    std::string longOption::getName() const {
      return name;
    }

    //---[ Parser ]---------------------
    parser::parser() {}

    option* parser::getShortOption(const std::string &opt) {
      if (opt.size() != 1) {
        return NULL;
      }
      const char optChar = opt[0];
      const int optCount = (int) options.size();
      for (int i = 0; i < optCount; ++i) {
        if (options[i].shortname == optChar) {
          return &(options[i]);
        }
      }
      return NULL;
    }

    option* parser::getOption(const std::string &opt) {
      const int optCount = (int) options.size();
      for (int i = 0; i < optCount; ++i) {
        if (options[i].name == opt) {
          return &(options[i]);
        }
      }
      return NULL;
    }

    bool parser::hasOptionalArg() {
      const int argumentCount = (int) arguments.size();
      return (argumentCount &&
              !arguments[argumentCount - 1].getIsRequired());
    }

    parser& parser::addArgument(const std::string &name_,
                                const std::string &description_,
                                const bool isRequired_) {

      OCCA_ERROR("Cannot add " << arguments[arguments.size() - 1]
                 << ", an optional argument has already been added\n",
                 !hasOptionalArg());

      option opt('\0', name_, description_);
      if (isRequired_) {
        opt = opt.isRequired();
      }
      arguments.push_back(opt);

      return *this;
    }

    parser& parser::addRepetitiveArgument(const std::string &name_,
                                          const std::string &description_,
                                          const bool isRequired_) {

      addArgument(name_, description_, isRequired_);
      hasRepetitiveArg = true;

      return *this;
    }

    parser& parser::addOption(const option &option) {
      options.push_back(option);
      return *this;
    }

    strVector parser::makeArgs(const int argc, const char **argv) {
      strVector args;
      for (int i = 0; i < argc; ++i) {
        args.push_back(argv[i]);
      }
      return args;
    }

    occa::json parser::parse(const int argc, const char **argv) {
      return parse(makeArgs(argc, argv));
    }

    occa::json parser::parse(const strVector &args) {
      occa::json parsedInfo(json::object_);
      const int argc = (int) args.size();

      // Set name to script name
      name = args[0];

      occa::json &jOrder     = parsedInfo["order"].asArray();
      occa::json &jOptions   = parsedInfo["options"].asObject();
      occa::json &jArguments = parsedInfo["arguments"].asArray();

      std::vector<option*> opts;
      option *opt = NULL;

      occa::json *optArgs = &jArguments;
      bool readingOpts = true;

      for (int i = 1; i < argc; ++i) {
        const std::string &arg_i = args[i];
        bool gotOpt = false;
        opts.clear();

        if (readingOpts) {
          if (startsWith(arg_i, "--")) {
            opt = getOption(arg_i.substr(2));
            gotOpt = true;
          } else if (startsWith(arg_i, "-")) {
            std::string args_i = arg_i.substr(1);
            const int shortArgs = (int) args_i.size();
            for (int j = 0; j < (shortArgs - 1); ++j) {
              opts.push_back(getShortOption(std::string(1, args_i[j])));
            }
            if (shortArgs) {
              opt = getShortOption(std::string(1, args_i[shortArgs - 1]));
            }
            gotOpt = true;
          } else {
            const int optArgCount = (int) optArgs->array().size();
            if (opt &&
                (opt->requiredArgs >= 0) &&
                (opt->requiredArgs <= optArgCount)) {
              opt = NULL;
              optArgs = &jArguments;
            }
            *optArgs += arg_i;
          }
        } else {
          jArguments += arg_i;
        }

        readingOpts = !jArguments.array().size();

        if (gotOpt) {
          if (((arg_i == "-h")     && !getShortOption("h")) ||
              ((arg_i == "--help") && !getShortOption("help"))) {

            printUsage(name);
            ::exit(0);
          }

          opts.push_back(opt);
          const int optCount = (int) opts.size();

          for (int j = 0; j < optCount; ++j) {
            if (!opts[j]) {
              std::cerr << "Unknown option: " << arg_i << '\n';
              printUsage(name, std::cerr);
              ::exit(1);
            }

            option &opt_j = *(opts[j]);
            jOrder += opt_j.name;

            // --foo a b       = [[a b]]
            // --foo a --foo b = [[a], [b]]
            json &argArrays = jOptions[opt_j.name].asArray();
            argArrays += json(json::array_);
            jsonArray &argArray = argArrays.array();
            optArgs = &(argArray[argArray.size() - 1]);
          }
        }
      }

      // Make sure required options were passed
      for (int i = 0; i < (int) options.size(); ++i) {
        option &opt_i = options[i];
        const bool hasOption = jOptions.has(opt_i.name);

        if (hasOption) {
          jsonArray optArgs_i = jOptions[opt_i.name].array();
          for (int j = 0; j < (int) optArgs_i.size(); ++j) {
            if (opt_i.requiredArgs != (int) optArgs_i[j].array().size()) {
              std::cerr << "Option " << opt_i << " requires "
                        << opt_i.requiredArgs << " argument"
                        << ((opt_i.requiredArgs > 1) ? "s\n" : "\n");
              printUsage(name, std::cerr);
              ::exit(1);
            }
          }
        } else if (opt_i.getIsRequired()) {
          std::cerr << "Option " << opt_i << " is required and missing\n";
          printUsage(name, std::cerr);
          ::exit(1);
        }
      }

      const int argCount = (int) jArguments.array().size();
      const int reqArgCount = (int) arguments.size() - hasOptionalArg();

      if (argCount < reqArgCount) {
        std::cerr << "Incorrect number of arguments\n";
        printUsage(name, std::cerr);
        ::exit(1);
      }

      return parsedInfo;
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
        printable::printDescription(out,
                                    0, MAX_NAME_COLUMN_WIDTH + MAX_DESC_COLUMN_WIDTH,
                                    description);
      } else {
        out << '\n';
      }
      out << '\n';

      printRequired(out);

      std::sort(options.begin(), options.end());
      printable::printEntries("Arguments", arguments, out);
      printable::printEntries("Options", options, out);
    }

    void parser::printRequired(std::ostream &out) {}

    //---[ Command ]--------------------
    command::command() :
      commandIsRequired(false),
      callback(NULL),
      runParent(NULL) {}

    command& command::withName(const std::string &name_) {
      name = name_;
      return *this;
    }

    command& command::withDescription(const std::string &description_) {
      description = description_;
      return *this;
    }

    command& command::withCallback(callback_t callback_) {
      callback = callback_;
      return *this;
    }

    command& command::withFunctionExpansion(std::string expansion) {
      expansionFunction = expansion;
      return *this;
    }

    int command::getCommandIdx(const std::string &name_) const {
      const int commandCount = (int) commands.size();
      for (int i = 0; i < commandCount; ++i) {
        const command &comm = commands[i];
        if (comm.name == name_) {
          return i;
        }
      }
      return -1;
    }

    const command* command::getCommand(const std::string &name_) const {
      const int idx = getCommandIdx(name_);
      return idx < 0 ? NULL : &commands[idx];
    }

    command* command::getCommand(const std::string &name_) {
      const int idx = getCommandIdx(name_);
      return idx < 0 ? NULL : &commands[idx];
    }

    void command::fillProgram(std::string &program) {
      if (runParent) {
        runParent->fillProgram(program);
        if (name.size()) {
          program += ' ';
          program += name;
        }
      } else {
        program = runArgs[0];
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
      printable::printEntries("Commands", commands, out);
    }

    command& command::requiresCommand() {
      commandIsRequired = true;
      return *this;
    }

    command& command::addCommand(const occa::cli::command &command_) {
      commands.push_back(command_);
      return *this;
    }

    void command::run(const int argc, const char **argv) {
      run(makeArgs(argc, argv));
    }

    void command::run(const strVector &args,
                      command *parent) {
      runParent = parent;
      runArgs = args;

      const bool hasCommands = commands.size();

      if (hasCommands) {
        addArgument("COMMAND",
                    "Command to run",
                    commandIsRequired);
      }

      json info = parse(args);

      json &jArguments = info["arguments"];
      strVector inputArgs = jArguments.getArray<std::string>();

      const int commandArg = arguments.size() - 1;
      std::string commandName;
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

        commandName = inputArgs[0];
        comm = getCommand(commandName);
      }

      if (callback && !callback(*this,
                                info["order"].array(),
                                info["options"].object(),
                                info["arguments"].array())) {
        printUsage(std::cerr);
        ::exit(1);
      }

      if (comm) {
        comm->run(inputArgs, this);
      } else if (commandIsRequired) {
        std::cerr << "Unknown command: " << commandName << '\n';
        printUsage(std::cerr);
        ::exit(1);
      }
    }

    void command::printBashAutocomplete(const std::string &funcPrefix) {
      const std::string funcName = funcPrefix + "_" + name;
      const bool isRoot = (funcPrefix.size() == 0);

      if (isRoot) {
        std::cout << io::read(env::OCCA_DIR + "scripts/autocomplete.sh");
      }

      std::cout << '\n'
                << funcName << "() {\n";
      // Setup global variables in the root command
      if (isRoot) {
        std::cout <<
          ("    __occa_debug_echo \"\"\n"
           "    __occa_debug_echo \"COMP_CWORD      : [${COMP_CWORD}]\"\n"
           "    __occa_debug_echo \"COMP_LINE       : [${COMP_LINE}]\"\n"
           "    __occa_debug_echo \"COMP_POINT      : [${COMP_POINT}]\"\n"
           "    __occa_debug_echo \"COMP_WORDBREAKS : [${COMP_WORDBREAKS}]\"\n"
           "    __occa_debug_echo \"COMP_WORDS      : [${COMP_WORDS}]\"\n"
           "\n"
           "    # Global variables\n"
           "    local command=(") << name << (")\n"
                                              "    local prevCommand=(") << name << (")\n"
                                                                                     "    local inputs=(\"${COMP_WORDS[@]:1}\")\n"
                                                                                     "    local nextInput=$(__occa_next_input)\n"
                                                                                     "    local options=()\n"
                                                                                     "    local flags=()\n"
                                                                                     "    local allUsedArgs=()\n"
                                                                                     "    local usedFlags=()\n"
                                                                                     "    local usedArgs=()\n"
                                                                                     "    local expansions=(args)\n"
                                                                                     "    local epansionFunction=\"\"\n"
                                                                                     "    local commandOptions=()\n"
                                                                                     "    local commandFlags=()\n"
                                                                                     "    local currentFlag=\"\"\n"
                                                                                     "    local compIsDone=false\n"
                                                                                     "\n"
                                                                                     "    # Real command info\n");
      }

      // Terminology mixup...
      // commands -> options
      // options  -> flags
      std::string optionsArray = "";
      std::string flagsArray   = "";
      for (int i = 0; i < (int) commands.size(); ++i) {
        if (i) {
          optionsArray += ' ';
        }
        optionsArray += "'" + commands[i].name + "'";
      }
      for (int i = 0; i < (int) options.size(); ++i) {
        option &opt = options[i];
        if (i) {
          flagsArray += ' ';
        }
        if (opt.shortname) {
          flagsArray += "'-";
          flagsArray += opt.shortname;
          flagsArray += "'";
          if (opt.name.size()) {
            flagsArray += ' ';
          }
        }
        if (opt.name.size()) {
          flagsArray += "'--" + opt.name + "'";
        }
      }
      std::cout << "    options=(" << optionsArray << ")\n"
                << "    flags=("   << flagsArray   << ")\n"
                << "\n"
                << "    __occa_init_command\n";

      if (commands.size()) {
        // If we have commands, always try to autocomplete
        std::cout << "    expansions=(args)\n"
                  << "    __occa_compgen\n";
      } else if (options.size()) {
        // If the next input is an unused flag, use the (args) expansion to
        //   continue the autocomplete
        std::cout <<
          ("    local unusedFlags=$(__occa_unused_flags)\n"
           "\n"
           "    if [ $(__occa_input_in \"${unusedFlags[@]}\") ]; then\n"
           "        expansions=(args)\n"
           "        __occa_compgen\n"
           "    else\n"
           "        __occa_autocomplete \"${unusedFlags[@]}\"\n"
           "        compIsDone=true\n"
           "    fi\n");
      } else {
        // Otherwise, use expansion type
        if (expansionFunction.size()) {
          std::cout << "    expansions=(func)\n"
                    << "    expansionFunction=\"" << expansionFunction << "\"";
        } else {
          std::cout << "    expansions=(file)\n";
        }
        std::cout << "    __occa_compgen\n";
      }

      std::cout << "}\n";

      for (int i = 0; i < (int) options.size(); ++i) {
        options[i].printBashAutocomplete(funcName);
      }

      for (int i = 0; i < (int) commands.size(); ++i) {
        commands[i].printBashAutocomplete(funcName);
      }

      if (isRoot) {
        std::cout << '\n'
                  << "complete -F " << funcName << " " << name << "\n";
      }
    }

    bool command::operator < (const command &comm) const {
      return name < comm.name;
    }
  }
}
