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
#ifndef OCCA_TOOLS_CLI_HEADER
#define OCCA_TOOLS_CLI_HEADER

#include <iostream>
#include <vector>

#include <occa/types.hpp>
#include <occa/tools/json.hpp>

namespace occa {
  namespace cli {
    namespace pretty {
      static const int COLUMN_SPACING = 3;
      static const int MAX_NAME_COLUMN_WIDTH = 30;
      static const int MAX_DESC_COLUMN_WIDTH = 50;

      // Printable entries
      template <class TM>
      void printEntries(const std::string &title,
                        const std::vector<TM> &entries,
                        std::ostream &out);

      void printDescription(std::ostream &out,
                            const std::string &description);

      void printDescription(std::ostream &out,
                            const int indent, const int width,
                            const std::string &description);
    }

    //---[ Printable ]------------------
    class printable {
    public:
      std::string name;
      std::string description;

      virtual std::string getPrintName() const = 0;
    };
    //==================================

    //---[ Option ]---------------------
    class option : public printable {
    public:
      class flags_t {
      public:
        static const int isRequired      = (1 << 0);
        static const int reusable        = (1 << 1);
        static const int stopsExpansion  = (1 << 2);
        static const int expandsFiles    = (1 << 3);
        static const int expandsFunction = (1 << 4);
      };

      char shortname;

      int flags;
      int requiredArgs;
      std::string expansionFunction;
      json defaultValue;

      option();

      option(const char shortname_,
             const std::string &name_,
             const std::string &description_);

      option(const std::string &name_,
             const std::string &description_);

      option isRequired();
      option reusable();
      option withArg();
      option withArgs(const int requiredArgs_);
      option withDefaultValue(const json &defaultValue_);

      option stopsExpansion();
      option expandsFiles();
      option expandsFunction(const std::string &function);

      bool getIsRequired();
      bool getReusable();
      bool hasDefaultValue();

      virtual std::string getPrintName() const;
      virtual std::string toString() const;

      void printBashAutocomplete(const std::string &funcPrefix);

      friend bool operator < (const option &l, const option &r);
      friend std::ostream& operator << (std::ostream &out, const option &opt);
    };
    //==================================

    //---[ Argument ]-------------------
    class argument: public option {
    public:
      argument();
      argument(const option &opt);

      argument(const std::string &name_,
               const std::string &description_);

      virtual std::string getPrintName() const;
      virtual std::string toString() const;
    };
    //==================================

    //---[ Parser ]---------------------
    class parser : public printable {
    public:
      std::vector<argument> arguments;
      std::vector<option> options;
      bool hasRepetitiveArg;

      parser();

      virtual std::string getPrintName() const;

      bool isLongOption(const std::string &arg);
      bool isShortOption(const std::string &arg);
      bool hasShortOption(const std::string &arg);
      bool isOption(const std::string &arg);

      option* getShortOption(const char opt,
                             const bool errorIfMissing = true);
      option* getLongOption(const std::string &opt,
                            const bool errorIfMissing = true);
      option* getOption(const std::string &arg,
                        const bool errorIfMissing = true);

      bool hasOptionalArg();

      parser& withDescription(const std::string &description_);

      parser& addArgument(const std::string &name_,
                          const std::string &description_,
                          const bool isRequired);

      parser& addRepetitiveArgument(const std::string &name_,
                                    const std::string &description_,
                                    const bool isRequired);

      parser& addOption(const option &option);

      strVector vectorizeArgs(const int argc, const char **argv);
      strVector splitShortOptionArgs(const strVector &args);

      occa::json parseArgs(const int argc, const char **argv);
      occa::json parseArgs(const strVector &args_);

      bool hasCustomHelpOption();
      void addHelpOption();
      void setOptionDefaults(occa::json &jOptions);

      virtual void printUsage(const std::string &program,
                              std::ostream &out = std::cout);

      virtual void printRequired(std::ostream &out);

      void fatalError(const std::string &message);

      void parseShortOptions(const std::string arg);
    };
    //==================================

    //---[ Command ]--------------------
    class command : public parser {
    public:
      typedef bool (*callback_t)(const occa::json &args);

      bool commandIsRequired;
      std::vector<command> commands;

      callback_t callback;
      std::string expansionFunction;

      command *runParent;
      strVector runArgs;

      command();

      command& withName(const std::string &name_);
      command& withDescription(const std::string &description_);
      command& withCallback(callback_t callback_);
      command& withFunctionExpansion(std::string expansion);

      int getCommandIdx(const std::string &name_) const;
      command* getCommand(const std::string &name_);
      const command* getCommand(const std::string &name_) const;

      void fillProgram(std::string &program);

      void printUsage(std::ostream &out = std::cout);

      virtual void printUsage(const std::string &program,
                              std::ostream &out = std::cout);

      virtual void printRequired(std::ostream &out);

      command& requiresCommand();
      command& addCommand(const occa::cli::command &command_);

      void run(const int argc, const char **argv);
      void run(const strVector &args,
               command *parent = NULL);

      void printBashAutocomplete(const std::string &funcPrefix="");

      bool operator < (const command &comm) const;
    };
    //==================================

    //---[ JSON ]-----------------------
    json parse(const int argc,
               const char **argv,
               const char *config);
    //==================================
  }
}

#include "cli.tpp"

#endif
