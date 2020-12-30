#ifndef OCCA_INTERNAL_UTILS_CLI_HEADER
#define OCCA_INTERNAL_UTILS_CLI_HEADER

#include <functional>
#include <iostream>
#include <vector>

#include <occa/types.hpp>
#include <occa/types/json.hpp>

namespace occa {
  namespace cli {
    static const std::string BASH_STOPS_EXPANSION = "stops-expansion";
    static const std::string BASH_EXPANDS_FILES = "expands-files";

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

      virtual ~printable() = 0;

      virtual std::string getPrintName() const = 0;
    };
    //==================================

    //---[ Option ]---------------------
    class option : public printable {
    public:
      typedef std::function<strVector (const json &args)> functionExpansionCallback;

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
      functionExpansionCallback expansionFunction;
      json defaultValue;

      option();

      option(const char shortname_,
             const std::string &name_,
             const std::string &description_);

      option(const std::string &name_,
             const std::string &description_);

      ~option();

      option isRequired(const bool required = true);
      option reusable();
      option withArg();
      option withArgs(const int requiredArgs_);
      option withDefaultValue(const json &defaultValue_);

      option stopsExpansion();
      option expandsFiles();
      option expandsFunction(functionExpansionCallback expansionFunction_);

      std::string getShortnameFlag() const;
      std::string getNameFlag() const;

      bool getIsRequired();
      bool getReusable();
      bool getStopsExpansion();
      bool getExpandsFiles();
      bool getExpandsFunction();
      bool hasDefaultValue();

      virtual std::string getPrintName() const;
      virtual std::string toString() const;

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

      ~argument();

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
      ~parser();

      virtual std::string getPrintName() const;

      static bool isLongOption(const std::string &arg);
      static bool isShortOption(const std::string &arg);
      static bool hasShortOption(const std::string &arg);
      static bool isOption(const std::string &arg);

      option* getShortOption(const char opt,
                             const bool errorIfMissing = true);
      option* getLongOption(const std::string &opt,
                            const bool errorIfMissing = true);
      option* getOption(const std::string &arg,
                        const bool errorIfMissing = true);

      bool hasOptionalArg();

      parser& withDescription(const std::string &description_);

      parser& addArgument(const argument &arg);

      parser& addOption(const option &option);

      strVector vectorizeArgs(const int argc, const char **argv);
      strVector splitShortOptionArgs(const strVector &args);

      occa::json parseArgs(const int argc, const char **argv);
      occa::json parseArgs(const strVector &args_,
                           const bool supressErrors = false);

      bool hasCustomHelpOption();
      void addHelpOption();
      void setOptionDefaults(occa::json &jOptions);

      virtual void printUsage(const std::string &program,
                              std::ostream &out = std::cerr);

      virtual void printRequired(std::ostream &out);

      void fatalError(const std::string &message);
    };
    //==================================

    //---[ Command ]--------------------
    class command : public parser {
    public:
      typedef std::function<bool (const occa::json &args)> commandCallback;

      mutable std::string commandPath;

      bool commandIsRequired;
      std::vector<command> commands;

      commandCallback callback;
      std::string expansionFunction;

      strVector runArgs;

      command();
      ~command();

      command& withName(const std::string &name_);
      command& withDescription(const std::string &description_);
      command& withCallback(commandCallback callback_);
      command& withFunctionExpansion(std::string expansion);

      command* getCommand(const std::string &name_);

      void fillProgram(std::string &program);

      void printUsage(std::ostream &out = std::cerr);

      virtual void printUsage(const std::string &program,
                              std::ostream &out = std::cerr);

      virtual void printRequired(std::ostream &out);

      command& requiresCommand();
      command& addCommand(const occa::cli::command &command_);
      void setCommandPath(const std::string &commandPath_) const;

      void run(const int argc, const char **argv);

      void run(const strVector &args);

      bool findCommandAndArguments(const strVector &shellArgs,
                                   command *&lastCommand,
                                   std::string &lastCommandName,
                                   json &lastCommandArgs,
                                   const bool supressErrors = false);

      void printBashAutocomplete(const std::string &fullBashCommand);

      void printBashSuggestions(const strVector &args);

      strVector getCommandBashSuggestions(const strVector &shellArgs,
                                          const json &args,
                                          const std::string &autocompleteArg);

      strVector stopBashAutocomplete();
      strVector getBashFileExpansion();

      strVector getCommandSuggestions();

      strVector getOptionFlagSuggestions(const strVector &usedOptions);

      strVector getOptionSuggestions(option &opt,
                                     const json &optArgs = json());

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
