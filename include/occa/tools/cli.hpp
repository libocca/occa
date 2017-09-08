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

#ifndef OCCA_TOOLS_ARGS_HEADER
#define OCCA_TOOLS_ARGS_HEADER

#include <iostream>
#include <vector>
#include <map>

#include "occa/types.hpp"
#include "occa/tools/json.hpp"

namespace occa {
  namespace cli {
    //---[ Printable ]------------------
    class printable {
    public:
      static const int COLUMN_SPACING = 3;
      static const int MAX_NAME_COLUMN_WIDTH = 30;
      static const int MAX_DESC_COLUMN_WIDTH = 50;

      std::string name;
      std::string description;

      printable();

      virtual std::string getName() const;

      template <class TM>
      static void printEntries(const std::string &title,
                               const std::vector<TM> &entries,
                               std::ostream &out) {
        if (!entries.size()) {
          return;
        }

        std::stringstream ss;

        out << title << ":\n";

        int nameColumnWidth = 0;

        const int entryCount = (int) entries.size();

        // Get maximum size needed to print name
        for (int i = 0; i < entryCount; ++i) {
          const TM &entry = entries[i];
          const int nameSize = (int) entry.getName().size();
          if (nameSize > nameColumnWidth) {
            nameColumnWidth = nameSize;
          }
        }
        nameColumnWidth += COLUMN_SPACING;
        if (nameColumnWidth > MAX_NAME_COLUMN_WIDTH) {
          nameColumnWidth = MAX_NAME_COLUMN_WIDTH;
        }

        for (int i = 0; i < entryCount; ++i) {
          const TM &entry = entries[i];
          ss << "  " << entry.getName();

          // If the line is larger than 'nameColumnWidth', start the
          //   description in the next line
          if ((int) ss.str().size() > (nameColumnWidth + COLUMN_SPACING)) {
            ss << '\n' << std::string(nameColumnWidth + COLUMN_SPACING, ' ');
          } else {
            ss << std::string(nameColumnWidth + COLUMN_SPACING - ss.str().size(), ' ');
          }

          out << ss.str();
          ss.str("");

          printDescription(out,
                           nameColumnWidth + COLUMN_SPACING,
                           MAX_DESC_COLUMN_WIDTH,
                           entry.description);
        }
        out << '\n';
      }

      static void printDescription(std::ostream &out,
                                   const int indent, const int width,
                                   const std::string &description_);
    };

    //---[ Option ]---------------------
    class option : public printable {
    public:
      char shortname;

      class flags_t {
      public:
        static const int isRequired      = (1 << 0);
        static const int reusable        = (1 << 1);
        static const int stopsExpansion  = (1 << 2);
        static const int expandsFiles    = (1 << 3);
        static const int expandsFunction = (1 << 4);
      };

      int flags;
      int requiredArgs;
      std::string expansionFunction;

      option();

      option(const char shortname_,
             const std::string &name_,
             const std::string &description_);

      option isRequired();
      option reusable();
      option withArgs(const int requiredArgs_);

      option stopsExpansion();
      option expandsFiles();
      option expandsFunction(const std::string &function);

      bool getIsRequired();

      virtual std::string getName() const;

      void printBashAutocomplete(const std::string &funcPrefix);

      friend bool operator < (const option &l, const option &r);
      friend std::ostream& operator << (std::ostream &out, const option &opt);
    };

    class longOption : public option {
    public:
      longOption();
      longOption(const option &opt);

      virtual std::string getName() const;
    };

    //---[ Parser ]---------------------
    class parser : public printable {
    public:
      std::vector<longOption> arguments;
      std::vector<option> options;
      bool hasRepetitiveArg;

      parser();

      option* getShortOption(const std::string &opt);
      option* getOption(const std::string &opt);

      bool hasOptionalArg();

      parser& addArgument(const std::string &name_,
                          const std::string &description_,
                          const bool isRequired);

      parser& addRepetitiveArgument(const std::string &name_,
                                    const std::string &description_,
                                    const bool isRequired);

      parser& addOption(const option &option);

      strVector_t makeArgs(const int argc, const char **argv);

      occa::json parse(const int argc, const char **argv);
      occa::json parse(const strVector_t &args);

      virtual void printUsage(const std::string &program,
                              std::ostream &out = std::cout);

      virtual void printRequired(std::ostream &out);
    };

    //---[ Command ]--------------------
    class command : public parser {
    public:
      typedef bool (*callback_t)(const occa::cli::command &command,
                                 jsonArray_t order,
                                 jsonObject_t options,
                                 jsonArray_t arguments);

      bool commandIsRequired;
      std::vector<command> commands;

      callback_t callback;
      std::string expansionFunction;

      command *runParent;
      strVector_t runArgs;

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
      void run(const strVector_t &args,
               command *parent = NULL);

      void printBashAutocomplete(const std::string &funcPrefix="");

      bool operator < (const command &comm) const;
    };
  }
}

#endif
