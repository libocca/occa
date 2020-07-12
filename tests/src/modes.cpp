#include <occa/tools/testing.hpp>

#include <occa/modes.hpp>

void testMode();

int main(const int argc, const char **argv) {
  testMode();

  return 0;
}

void testMode() {
  occa::mode_t *serialMode = occa::getMode("mode: 'Serial'");
  ASSERT_NEQ((void*) serialMode,
             (void*) NULL);

  ASSERT_EQ(occa::getMode(""),
            serialMode);

  ASSERT_EQ(occa::getMode("mode: 'Foo'"),
            serialMode);
}
