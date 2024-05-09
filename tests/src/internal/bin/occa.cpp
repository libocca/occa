#include <algorithm>
#include <iostream>
#include <sstream>

#include <occa.hpp>

#include <occa/internal/utils/cli.hpp>
#include <occa/internal/utils/string.hpp>
#include <occa/internal/utils/testing.hpp>
#include <occa/internal/modes.hpp>
#include <occa/internal/bin/occa.hpp>

std::stringstream ss;

void saveOutput(const char *str) {
  ss << str;
}

std::string getAutocompleteOutput(occa::cli::command &occaCommand,
                                  const std::string &commandLine) {
  ss.str("");

  occa::strVector commandLineArgs = occa::split(
    occa::strip(commandLine),
    ' '
  );

  // Ending with a space is a simple way to notify
  // there's an empty arg afterwards
  if (occa::endsWith(commandLine, " ")) {
    commandLineArgs.push_back("");
  }

  occaCommand.printBashSuggestions(commandLineArgs);

  std::string suggestions = ss.str();

  std::replace(suggestions.begin(), suggestions.end(),
               '\n', ' ');

  return occa::strip(suggestions);
}

std::string getModes() {
  occa::strVector modesVec;
  for (auto &it : occa::getModeMap()) {
    modesVec.push_back(it.second->name());
  }
  std::sort(modesVec.begin(), modesVec.end());

  std::string modes;
  for (auto &mode : modesVec) {
    if (modes.size()) {
      modes += ' ';
    }
    modes += mode;
  }

  return modes;
}

#define ASSERT_AUTOCOMPLETE_EQ(COMMAND_LINE, EXPECTED_OUTPUT) \
  ASSERT_EQ(                                                  \
    getAutocompleteOutput(occaCommand, COMMAND_LINE),         \
    EXPECTED_OUTPUT                                           \
  )

int main(const int argc, const char **argv) {
  occa::cli::command occaCommand = occa::bin::buildOccaCommand();

  occa::io::stdout.setOverride(saveOutput);

  const std::string commands = "autocomplete clear compile env info modes translate version";
  const std::string helpOptions = "--help -h";

  const std::string modeSuggetions = getModes();

  //---[ OCCA ]----------------------------
  ASSERT_AUTOCOMPLETE_EQ(
    "occa",
    commands
  );

  ASSERT_AUTOCOMPLETE_EQ(
    "occa a",
    commands
  );

  ASSERT_AUTOCOMPLETE_EQ(
    "occa c",
    commands
  );

  ASSERT_AUTOCOMPLETE_EQ(
    "occa cl",
    commands
  );

  ASSERT_AUTOCOMPLETE_EQ(
    "occa -",
    helpOptions
  );

  ASSERT_AUTOCOMPLETE_EQ(
    "occa --",
    helpOptions
  );

  //---[ Autocomplete ]--------------------
  ASSERT_AUTOCOMPLETE_EQ(
    "occa autocomplete b",
    "bash"
  );

  ASSERT_AUTOCOMPLETE_EQ(
    "occa autocomplete bash",
    "bash"
  );

  ASSERT_AUTOCOMPLETE_EQ(
    "occa autocomplete bash  ",
    occa::cli::BASH_STOPS_EXPANSION
  );

  //---[ Clear ]---------------------------
  const std::string clearOptions = "--all --help --kernels --locks --yes -a -h -l -y";
  ASSERT_AUTOCOMPLETE_EQ(
    "occa clear -",
    clearOptions
  );

  ASSERT_AUTOCOMPLETE_EQ(
    "occa clear --a",
    clearOptions
  );

  ASSERT_AUTOCOMPLETE_EQ(
    "occa clear --all",
    clearOptions
  );

  ASSERT_AUTOCOMPLETE_EQ(
    "occa clear --locks",
    clearOptions
  );

  ASSERT_AUTOCOMPLETE_EQ(
    "occa clear --locks -l",
    "--all --help --kernels --yes -a -h -y"
  );

  //---[ Env ]-----------------------------
  ASSERT_AUTOCOMPLETE_EQ(
    "occa env -",
    helpOptions
  );

  ASSERT_AUTOCOMPLETE_EQ(
    "occa env --h",
    helpOptions
  );

  //---[ Info ]----------------------------
  ASSERT_AUTOCOMPLETE_EQ(
    "occa info -",
    helpOptions
  );

  ASSERT_AUTOCOMPLETE_EQ(
    "occa info --h",
    helpOptions
  );

  //---[ Compile ]------------------
  const std::string compileOptions = (
    "--define --device-props --help --include-path --kernel-props --transpiler-version -D -I -d -h -k -t"
  );

  ASSERT_AUTOCOMPLETE_EQ(
    "occa compile  ",
    occa::cli::BASH_EXPANDS_FILES
  );

  ASSERT_AUTOCOMPLETE_EQ(
    "occa compile dir1/",
    occa::cli::BASH_EXPANDS_FILES
  );

  ASSERT_AUTOCOMPLETE_EQ(
    "occa compile file1",
    occa::cli::BASH_EXPANDS_FILES
  );

  ASSERT_AUTOCOMPLETE_EQ(
    "occa compile file1 -",
    compileOptions
  );

  ASSERT_AUTOCOMPLETE_EQ(
    "occa compile file1 --",
    compileOptions
  );

  //---[ Translate ]------------------
  const std::string translateOptions = (
    "--define --help --include-path --kernel-props --launcher --mode --transpiler-version --verbose -D -I -h -k -l -m -t -v"
  );

  ASSERT_AUTOCOMPLETE_EQ(
    "occa translate  ",
    occa::cli::BASH_EXPANDS_FILES
  );

  ASSERT_AUTOCOMPLETE_EQ(
    "occa translate dir1/",
    occa::cli::BASH_EXPANDS_FILES
  );

  ASSERT_AUTOCOMPLETE_EQ(
    "occa translate file1",
    occa::cli::BASH_EXPANDS_FILES
  );

  ASSERT_AUTOCOMPLETE_EQ(
    "occa translate file1 -",
    translateOptions
  );

  ASSERT_AUTOCOMPLETE_EQ(
    "occa translate file1 --",
    translateOptions
  );

  ASSERT_AUTOCOMPLETE_EQ(
    "occa translate file1 --mode ",
    modeSuggetions
  );

  //---[ Modes ]---------------------------
  ASSERT_AUTOCOMPLETE_EQ(
    "occa modes -",
    helpOptions
  );

  ASSERT_AUTOCOMPLETE_EQ(
    "occa modes --h",
    helpOptions
  );

  //---[ Version ]-------------------------
  ASSERT_AUTOCOMPLETE_EQ(
    "occa version -",
    "--help --okl -h"
  );

  ASSERT_AUTOCOMPLETE_EQ(
    "occa version --okl",
    "--help --okl -h"
  );

  return 0;
}
