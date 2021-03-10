#include "utils.hpp"

void testTypeMethods();

int main(const int argc, const char **argv) {
  setupParser();

  testTypeMethods();

  return 0;
}

void testTypeMethods() {
  setSource("int a = 0;");
  setSource("const int *a = 0;");

  // Make sure we can handle [long] and [long long]
  setSource("long a = 0;");
  setSource("const long a = 0;");

  setSource("long long a = 0;");
  setSource("const long long *a = 0;");

  // Make sure we load structs as structs (use long as a dummy known type)
  setSource("struct long a;");
  setSource("struct long a = 0;");
}
