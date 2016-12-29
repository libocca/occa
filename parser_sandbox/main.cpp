#include <iostream>

#if 0
#include "occa.hpp"
#include "occa/tools/io.hpp"
#include "occa/parser/tools.hpp"

#include "tools.hpp"
#include "preprocessor.hpp"
#endif

#include "primitive.hpp"

int main(int argc, char **argv) {
  primitive a = "15.1230093402";
  std::cout
    << "a = " << a << '\n';

#if 0
  std::string content = occa::io::read("cleanTest.c");
  const char *c = content.c_str();
  std::string processedContent;

  preprocessor_t preprocessor;
  preprocessor.process(c, processedContent);

  std::cout << processedContent << '\n';
#endif
  return 0;
}
