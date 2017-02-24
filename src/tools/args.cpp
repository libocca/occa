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

#include <algorithm>
#include <iomanip>
#include <sstream>

#include "occa/tools/args.hpp"
#include "occa/tools/string.hpp"
#include "occa/tools/lex.hpp"

namespace occa {
  namespace args {
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
    option::option() {}

    option::option(const std::string &name_,
                   const std::string &description_,
                   const int args_,
                   const bool isRequired_) :
      shortname(0),
      args(args_),
      isRequired(isRequired_) {

      name = name_;
      description = description_;
    }

    option::option(const char shortname_,
                   const std::string &name_,
                   const std::string &description_,
                   const int args_,
                   const bool isRequired_) :
      shortname(shortname_),
      args(args_),
      isRequired(isRequired_) {

      name = name_;
      description = description_;
    }

    std::string option::getName() const {
      std::string ret;
      if (shortname) {
        ret += '-';
        ret += shortname;
        ret += ", --";
        ret += name;
      } else {
        ret += "    --";
        ret += name;
      }
      return ret;
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
      args = opt.args;
      isRequired = opt.isRequired;
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

    parser& parser::withDescription(const std::string &description_) {
      description = description_;
      return *this;
    }

    bool parser::hasOptionalArg() {
      const int argumentCount = (int) arguments.size();
      return (argumentCount &&
              !arguments[argumentCount - 1].isRequired);
    }

    parser& parser::addArgument(const std::string &name_,
                                const std::string &description_,
                                const bool isRequired_) {

      OCCA_ERROR("Cannot add " << arguments[arguments.size() - 1]
                 << ", an optional argument has already been added\n",
                 !hasOptionalArg());

      arguments.push_back(option(name_, description_,
                                 0, isRequired_));

      return *this;
    }

    parser& parser::addRepetitiveArgument(const std::string &name_,
                                          const std::string &description_,
                                          const bool isRequired_) {

      addArgument(name_, description_, isRequired_);
      hasRepetitiveArg = true;

      return *this;
    }

    parser& parser::addOption(const std::string &name_,
                              const std::string &description_,
                              const int args,
                              const bool isRequired_) {

      options.push_back(option(name_, description_,
                               args, isRequired_));
      return *this;
    }

    parser& parser::addOption(const char shortname_,
                              const std::string &name_,
                              const std::string &description_,
                              const int args,
                           const bool isRequired_) {

      options.push_back(option(shortname_, name_, description_,
                               args, isRequired_));
      return *this;
    }

    strVector_t parser::makeArgs(const int argc, const char **argv) {
      strVector_t args;
      for (int i = 0; i < argc; ++i) {
        args.push_back(argv[i]);
      }
      return args;
    }

    occa::json parser::parse(const int argc, const char **argv) {
      return parse(makeArgs(argc, argv));
    }

    occa::json parser::parse(const strVector_t &args) {
      occa::json parsedInfo(json::object_);
      const int argc = (int) args.size();

      occa::json &jOrder = parsedInfo["order"].asArray();
      occa::json &jOptions = parsedInfo["options"].asObject();
      occa::json &jArguments = parsedInfo["arguments"].asArray();

      option *opt = NULL;
      occa::json *optArgs = &jArguments;
      bool readingOpts = true;

      for (int i = 1; i < argc; ++i) {
        const std::string &arg_i = args[i];
        bool gotOpt = false;

        if (readingOpts) {
          if (startsWith(arg_i, "--")) {
            opt = getOption(arg_i.substr(2));
            gotOpt = true;
          } else if (startsWith(arg_i, "-")) {
            opt = getShortOption(arg_i.substr(1));
            gotOpt = true;
          } else {
            const int optArgCount = (int) optArgs->array().size();
            if (opt && opt->args <= optArgCount) {
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

            printUsage(args[0]);
            ::exit(0);
          }

          if (opt == NULL) {
            std::cerr << "Unknown option: " << arg_i << '\n';
            printUsage(args[0], std::cerr);
            ::exit(1);
          }

          jOrder += opt->name;

          // --foo a b       = [[a b]]
          // --foo a --foo b = [[a], [b]]
          json &argArrays = jOptions[opt->name].asArray();
          argArrays += json(json::array_);
          jsonArray_t &argArray = argArrays.array();
          optArgs = &(argArray[argArray.size() - 1]);
        }
      }

      // Make sure required options were passed
      for (int i = 0; i < (int) options.size(); ++i) {
        option &opt_i = options[i];
        const bool hasOption = jOptions.has(opt_i.name);

        if (hasOption) {
          jsonArray_t optArgs_i = jOptions[opt_i.name].array();
          for (int j = 0; j < (int) optArgs_i.size(); ++j) {
            if (opt_i.args != (int) optArgs_i[j].array().size()) {
              std::cerr << "Option " << opt_i << " is required and missing\n";
              printUsage(args[0], std::cerr);
              ::exit(1);
            }
          }
        } else if (opt_i.isRequired) {
          std::cerr << "Option " << opt_i << " is required and missing\n";
          printUsage(args[0], std::cerr);
          ::exit(1);
        }
      }

      const int argCount = (int) jArguments.array().size();
      const int reqArgCount = (int) arguments.size() - hasOptionalArg();

      if (argCount < reqArgCount) {
        std::cerr << "Incorrect number of arguments\n";
        printUsage(args[0], std::cerr);
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
        if (argument.isRequired) {
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
    command::command() {}

    command& command::withName(const std::string &name_) {
      name = name_;
      return *this;
    }

    command& command::withCallback(callback_t callback_) {
      callback = callback_;
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
      } else {
        program = runArgs[0];
      }

      if (name.size()) {
        program += ' ';
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
      printable::printEntries("Commands", commands, out);
    }

    command& command::requiresCommand() {
      commandIsRequired = true;
      return *this;
    }

    command& command::addCommand(const occa::args::command &command_) {
      commands.push_back(command_);
      return *this;
    }

    void command::run(const int argc, const char **argv) {
      run(makeArgs(argc, argv));
    }

    void command::run(const strVector_t &args,
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
      strVector_t inputArgs = jArguments.getArray<std::string>();

      const int commandArg = arguments.size() - 1;
      std::string commandName;
      command *comm = NULL;

      // Modify arguments and find command
      if (hasCommands &&
          inputArgs.size() &&
          commandArg < (int) inputArgs.size()) {

        // Remove command arguments
        jsonArray_t &jArgArray = jArguments.array();
        jArgArray = jsonArray_t(jArgArray.begin(),
                                jArgArray.begin() + commandArg + 1);

        // Extract command arguments
        inputArgs = strVector_t(inputArgs.begin() + commandArg,
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

    bool command::operator < (const command &comm) const {
      return name < comm.name;
    }
  }
}
