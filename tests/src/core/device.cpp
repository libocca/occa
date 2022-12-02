#include <occa.hpp>
#include <occa/internal/utils/testing.hpp>

void testProperties();
void testWrapMemory();
void testUnwrap();

int main(const int argc, const char **argv) {
  testProperties();
  testWrapMemory();
  testUnwrap();

  return 0;
}

void testProperties() {
  occa::device device;

  device.setup({
    {"mode", "Serial"}
  });
  ASSERT_EQ(
    "Serial",
    (std::string) device.properties()["mode"]
  );

  device.setup({
    {"mode", "Serial"},
    {"one", 1}
  });
  ASSERT_EQ(
    1,
    (int) device.properties()["one"]
  );

  // Object and mode overrides
  device.setup({
    {"mode", "Serial"}
  });
  ASSERT_EQ(
    "Serial",
    (std::string) device.kernelProperties()["mode"]
  );

  device.setup({
    {"mode", "Serial"},
    {"kernel", {
      {"one", 1}
    }}
  });
  ASSERT_EQ(
    1,
    (int) device.kernelProperties()["one"]
  );

  device.setup({
    {"mode", "Serial"},
    {"one", 2},
    {"kernel", {
      {"one", 1}
    }}
  });
  ASSERT_EQ(
    1,
    (int) device.kernelProperties()["one"]
  );

  device.setup({
    {"mode", "Serial"},
    {"one", 3},
    {"kernel", {
      {"one", 2},
      {"modes", {
        {"Serial", {
          {"one", 1}
        }}
      }}
    }}
  });
  ASSERT_EQ(
    1,
    (int) device.kernelProperties()["one"]
  );

  device.setup({
    {"mode", "Serial"},
    {"one", 3},
    {"modes", {
      {"one", 2},
      {"Serial", {
          {"kernel", {
            {"one", 1}
        }}
      }}
    }}
  });
  ASSERT_EQ(
    1,
    (int) device.kernelProperties()["one"]
  );
}

void testWrapMemory() {
  occa::device device({
    {"mode", "Serial"}
  });

  const occa::udim_t bytes = 1 * sizeof(int);
  int value = 4660;
  int *hostPtr = &value;

  occa::memory mem = device.wrapMemory((void*) hostPtr, bytes);
  ASSERT_EQ(mem.ptr<int>()[0], value);
  ASSERT_EQ(mem.ptr<int>(), hostPtr);
  ASSERT_EQ((int) mem.length<int>(), 1);

  mem = device.wrapMemory(hostPtr, 1);
  ASSERT_EQ(mem.ptr<int>()[0], value);
  ASSERT_EQ(mem.ptr<int>(), hostPtr);
  ASSERT_EQ((int) mem.length<int>(), 1);

  mem = device.wrapMemory(hostPtr, 1, {{"use_host_pointer", false}});
  ASSERT_EQ(mem.ptr<int>()[0], value);
  ASSERT_EQ(mem.ptr<int>(), hostPtr);
  ASSERT_EQ((int) mem.length<int>(), 1);
}

void testUnwrap() {
  occa::device device({
    {"mode","Serial"}
  });

  // Unwrapping a serial mode device is undefined
  ASSERT_THROW(occa::unwrap(device););
}
