#include <occa.hpp>
#include <occa/tools/testing.hpp>

void testProperties();

int main(const int argc, const char **argv) {
  testProperties();

  return 0;
}

void testProperties() {
  occa::device device;

  device.setup(
    "mode: 'Serial'"
  );
  ASSERT_EQ(
    "Serial",
    (std::string) device.properties()["mode"]
  );

  device.setup(
    "mode: 'Serial',"
    "one: 1"
  );
  ASSERT_EQ(
    1,
    (int) device.properties()["one"]
  );

  // Object and mode overrides
  device.setup(
    "mode: 'Serial'"
  );
  ASSERT_EQ(
    "Serial",
    (std::string) device.kernelProperties()["mode"]
  );

  device.setup(
    "mode: 'Serial',"
    "kernel: {"
    "  one: 1,"
    "}"
  );
  ASSERT_EQ(
    1,
    (int) device.kernelProperties()["one"]
  );

  device.setup(
    "mode: 'Serial',"
    "one: 2,"
    "kernel: {"
    "  one: 1,"
    "}"
  );
  ASSERT_EQ(
    1,
    (int) device.kernelProperties()["one"]
  );

  device.setup(
    "mode: 'Serial',"
    "one: 3,"
    "kernel: {"
    "  one: 2,"
    "  modes: {"
    "    Serial: {"
    "      one: 1,"
    "    },"
    "  },"
    "}"
  );
  ASSERT_EQ(
    1,
    (int) device.kernelProperties()["one"]
  );

  device.setup(
    "mode: 'Serial',"
    "one: 3,"
    "kernel: {"
    "  one: 2,"
    "},"
    "modes: {"
    "  Serial: {"
    "    kernel: {"
    "      one: 1,"
    "    },"
    "  },"
    "}"
  );
  ASSERT_EQ(
    1,
    (int) device.kernelProperties()["one"]
  );
}
