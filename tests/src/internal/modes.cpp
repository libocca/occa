#include <occa/internal/utils/testing.hpp>

#include <occa/internal/modes.hpp>

void testModeByName();
void testModeByProps();

int main(const int argc, const char **argv) {
  testModeByName();
  testModeByProps();

  return 0;
}

void testModeByName() {
  occa::mode_t *serialMode = occa::getMode("Serial");
  ASSERT_NEQ((void*) serialMode,
             (void*) NULL);

  ASSERT_EQ((void*) serialMode,
            (void*) occa::getMode("serial"));

  ASSERT_EQ((void*) occa::getMode(""),
            (void*) NULL);

  ASSERT_EQ((void*) occa::getMode("Foo"),
            (void*) NULL);
}

void testModeByProps() {
  occa::mode_t *serialMode = occa::getModeFromProps({
    {"mode", "Serial"}
  });

  ASSERT_NEQ((void*) serialMode,
             (void*) NULL);

  ASSERT_EQ((void*) serialMode,
            (void*) occa::getModeFromProps({{"mode", "Serial"}}));

  ASSERT_EQ(occa::getModeFromProps(""),
            serialMode);

  ASSERT_EQ(occa::getModeFromProps({{"mode", "Foo"}}),
            serialMode);
}
