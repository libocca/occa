#include <occa/internal/bin/occa.hpp>

int main(const int argc, const char **argv) {
  occa::bin::buildOccaCommand().run(argc, argv);

  return 0;
}
