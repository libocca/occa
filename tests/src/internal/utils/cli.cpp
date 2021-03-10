#include <iostream>
#include <sstream>
#include <occa/defines.hpp>
#include <occa/internal/utils/cli.hpp>
#include <occa/internal/utils/testing.hpp>

void testPretty();
void testOption();
void testArgument();
void testParser();
void testCommand();

int main(const int argc, const char **argv) {
  testPretty();
  testOption();
  testArgument();
  testParser();
  testCommand();

  return 0;
}

class DummyPrintable : public occa::cli::printable {
public:
  DummyPrintable(const std::string &name_,
                 const std::string &description_) {
    name = name_;
    description = description_;
  }

  std::string getPrintName() const {
    return name;
  }
};

void testPretty() {
  std::stringstream ss;
  std::vector<DummyPrintable> entries;

  occa::cli::pretty::printEntries("title", entries, ss);
  ASSERT_EQ("", ss.str());

  entries.push_back(DummyPrintable("1"  , "one"));
  entries.push_back(DummyPrintable("22" , "two"));
  entries.push_back(DummyPrintable("333", "three"));

  occa::cli::pretty::printEntries("title", entries, ss);
  ASSERT_EQ(
    "title:\n"
    "  1      one\n"
    "  22     two\n"
    "  333    three\n\n",
    ss.str());
  ss.str("");
  entries.clear();

  occa::cli::pretty::printDescription(ss, 0, 1, "a b c d");
  ASSERT_EQ(
    "\n"
    "a\n"
    "b\n"
    "c\n"
    "d\n",
    ss.str());
  ss.str("");

  occa::cli::pretty::printDescription(ss, 3, 1, "a b c d");
  ASSERT_EQ(
    "\n"
    "   a\n"
    "   b\n"
    "   c\n"
    "   d\n",
    ss.str());
  ss.str("");
}

void testOption() {
  occa::cli::option emptyOpt;
  occa::cli::option shortOpt('a', "aaa", "aaaahhhhhhhhh!!!!");
  occa::cli::option opt("ooo", "ooohhhhhhhhhhhh");

  ASSERT_FALSE(shortOpt.getIsRequired());
  ASSERT_FALSE(shortOpt.getReusable());
  ASSERT_FALSE(shortOpt.hasDefaultValue());

  // Test required
  shortOpt.isRequired();
  ASSERT_FALSE(shortOpt.getIsRequired());

  shortOpt = shortOpt.isRequired();
  ASSERT_TRUE(shortOpt.getIsRequired());
  ASSERT_FALSE(shortOpt.getReusable());
  ASSERT_FALSE(shortOpt.hasDefaultValue());

  // Test reusable
  shortOpt.reusable();
  ASSERT_FALSE(shortOpt.getReusable());

  shortOpt = shortOpt.reusable();
  ASSERT_TRUE(shortOpt.getIsRequired());
  ASSERT_TRUE(shortOpt.getReusable());
  ASSERT_FALSE(shortOpt.hasDefaultValue());

  // Test withDefaultValue
  shortOpt.withDefaultValue("null");
  ASSERT_FALSE(shortOpt.hasDefaultValue());

  shortOpt = shortOpt.withDefaultValue("null");
  ASSERT_TRUE(shortOpt.getIsRequired());
  ASSERT_TRUE(shortOpt.getReusable());
  ASSERT_TRUE(shortOpt.hasDefaultValue());

  // Test print names
  ASSERT_EQ(
    "-a, --aaa",
    shortOpt.getPrintName()
  );
  ASSERT_EQ(
    "    --ooo",
    opt.getPrintName()
  );

  ASSERT_EQ(
    "-a, --aaa",
    shortOpt.toString()
  );
  ASSERT_EQ(
    "--ooo",
    opt.toString()
  );

  std::stringstream ss;
  ss << shortOpt;
  ASSERT_EQ(
    "-a, --aaa",
    ss.str()
  );
  ss.str("");
  ss << opt;
  ASSERT_EQ(
    "--ooo",
    ss.str()
  );

  // Operators
  ASSERT_TRUE(shortOpt < opt);
  ASSERT_FALSE(opt < shortOpt);
}

void testArgument() {
  occa::cli::argument emptyArg;
  occa::cli::argument arg("name", "description");
  occa::cli::argument arg2(arg);

  ASSERT_EQ(
    "name",
    arg.getPrintName()
  );
  ASSERT_EQ(
    "name",
    arg2.getPrintName()
  );

  ASSERT_EQ(
    "name",
    arg.toString()
  );
  ASSERT_EQ(
    "name",
    arg2.toString()
  );
}

void testParser() {
  occa::cli::parser parser;

  ASSERT_TRUE(parser.isLongOption("--a"));
  ASSERT_FALSE(parser.isLongOption("--"));
  ASSERT_FALSE(parser.isLongOption("-a"));
  ASSERT_FALSE(parser.isLongOption("---a"));

  ASSERT_TRUE(parser.isShortOption("-a"));
  ASSERT_FALSE(parser.isShortOption("-"));
  ASSERT_FALSE(parser.isShortOption("--a"));
  ASSERT_FALSE(parser.isShortOption("a"));

  ASSERT_TRUE(parser.isOption("--a"));
  ASSERT_TRUE(parser.isOption("-a"));
  ASSERT_FALSE(parser.isOption("--"));
  ASSERT_FALSE(parser.isOption("-"));
  ASSERT_FALSE(parser.isOption("a"));
  ASSERT_FALSE(parser.isOption("---a"));

  const std::string description = "Example adding two vectors";
  parser
    .withDescription(description)
    .addOption(
      occa::cli::option('d', "device",
                        "Device properties (default: \"{mode: 'Serial'}\")")
      .withArg()
      .withDefaultValue("{mode: 'Serial'}")
    )
    .addOption(
      occa::cli::option('v', "verbose",
                        "Compile kernels in verbose mode")
    );

  ASSERT_EQ(parser.description, description);

  // Test option retrieval
  occa::cli::option *opt = parser.getShortOption('a', false);
  ASSERT_EQ((void*) opt, (void*) NULL);

  opt = parser.getShortOption('d', false);
  ASSERT_NEQ((void*) opt, (void*) NULL);
  ASSERT_EQ(opt->toString(), "-d, --device");

  // Test custom help
  ASSERT_FALSE(parser.hasCustomHelpOption());

  parser.addOption(
    occa::cli::option('h', "not-help", "This is not help")
  );
  ASSERT_FALSE(parser.hasCustomHelpOption());

  parser.addOption(
    occa::cli::option("help", "TODO: help goes here")
  );
  ASSERT_TRUE(parser.hasCustomHelpOption());
}

void testCommand() {
}
