#include <occa.hpp>
#include <occa/tools/testing.hpp>

void testMalloc();
void testCpuWrapMemory();

int main(const int argc, const char **argv) {
  testMalloc();
  testCpuWrapMemory();

  return 0;
}

void testMalloc() {
  const occa::udim_t bytes = 1 * sizeof(int);
  int value = 4660;
  int *hostPtr = &value;

  occa::memory mem = occa::malloc(bytes);
  ASSERT_NEQ(mem.ptr(), (void*) NULL);

  mem = occa::malloc(bytes, hostPtr);
  ASSERT_EQ(((int*) mem.ptr())[0], value);
  ASSERT_NEQ(mem.ptr<int>(), hostPtr);

  mem = occa::malloc(bytes, hostPtr, "use_host_pointer: true");
  ASSERT_EQ(mem.ptr<int>()[0], value);
  ASSERT_EQ(mem.ptr<int>(), hostPtr);

  occa::setDevice(
    "mode: 'Serial',"
    "memory: {"
    "  use_host_pointer: true,"
    "}"
  );

  mem = occa::malloc(bytes, hostPtr);
  ASSERT_EQ(((int*) mem.ptr())[0], value);
  ASSERT_EQ(mem.ptr<int>(), hostPtr);

  mem = occa::malloc(bytes, hostPtr, "use_host_pointer: false");
  ASSERT_EQ(mem.ptr<int>()[0], value);
  ASSERT_NEQ(mem.ptr<int>(), hostPtr);
}

void testCpuWrapMemory() {
  const occa::udim_t bytes = 1 * sizeof(int);
  int value = 4660;
  int *hostPtr = &value;

  occa::memory mem = occa::cpu::wrapMemory(hostPtr, bytes);
  ASSERT_EQ(mem.ptr<int>()[0], value);
  ASSERT_EQ(mem.ptr<int>(), hostPtr);

  mem = occa::cpu::wrapMemory(hostPtr, bytes, "use_host_pointer: false");
  ASSERT_EQ(mem.ptr<int>()[0], value);
  ASSERT_NEQ(mem.ptr<int>(), hostPtr);
}
