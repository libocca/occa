#include <occa/internal/utils/testing.hpp>

#include <occa/internal/lang/file.hpp>

using namespace occa::lang;

int main(const int argc, const char **argv) {
  fileOrigin orig, orig2;
  std::string line = "#define foo bar";
  const char *c_line = line.c_str();
  const int foo = 8;
  const int bar = 12;
  orig.position = filePosition(10,
                               c_line,
                               c_line + foo,
                               c_line + foo + 3);
  orig.push(true,
            originSource::string,
            filePosition(20,
                         c_line,
                         c_line + bar,
                         c_line + bar + 3));
  orig2 = orig;
  orig.push(false,
            originSource::string,
            filePosition(30,
                         c_line,
                         c_line + foo,
                         c_line + foo + 3));
  orig2.push(false,
             originSource::string,
             filePosition(40,
                          c_line,
                          c_line + bar,
                          c_line + bar + 3));

  orig.print(occa::io::stdout);
  std::cout << "Test message\n\n";

  orig2.print(occa::io::stdout);
  std::cout << "Test message 2\n";

  return 0;
}
