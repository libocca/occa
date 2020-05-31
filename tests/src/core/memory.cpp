#include <occa.hpp>
#include <occa/tools/testing.hpp>

void testMalloc();
void testCpuWrapMemory();
void testSlice();

int main(const int argc, const char **argv) {
  testMalloc();
  testCpuWrapMemory();
  testSlice();

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

void testSlice() {
#define ASSERT_SAME_SIZE(a, b) \
  ASSERT_EQ((size_t) (a), (size_t) (b))

  float *data = new float[10];
  for (int i = 0; i < 10; ++i) {
    data[i] = i;
  }

  occa::device device("mode: 'Serial'");
  ASSERT_SAME_SIZE(device.memoryAllocated(), 0);

  {
    occa::memory mem = device.malloc<float>(10, data);
    ASSERT_SAME_SIZE(device.memoryAllocated(), 10 * sizeof(float));
    {
      occa::memory half1 = mem.slice(0, 5);
      occa::memory half2 = mem.slice(5);
    }

    ASSERT_SAME_SIZE(device.memoryAllocated(), 10 * sizeof(float));
    {
      occa::memory half1 = mem.slice(0, 5);
      occa::memory half2 = mem.slice(5);

      half2.copyTo(data);
      for (int i = 0; i < 5; ++i) {
        ASSERT_EQ(static_cast<int>(data[i]), 5 + i);
      }

      half1.copyTo(data);
      for (int i = 0; i < 5; ++i) {
        ASSERT_EQ(static_cast<int>(data[i]), i);
      }
    }

    ASSERT_SAME_SIZE(device.memoryAllocated(), 10 * sizeof(float));
    {
      occa::memory half1 = mem + 0;
      occa::memory tmp   = half1 + 2;
      occa::memory half2 = tmp + 3;

      half2.copyTo(data);
      for (int i = 0; i < 5; ++i) {
        ASSERT_EQ(static_cast<int>(data[i]), 5 + i);
      }

      half1.copyTo(data);
      for (int i = 0; i < 5; ++i) {
        ASSERT_EQ(static_cast<int>(data[i]), i);
      }
    }

    ASSERT_SAME_SIZE(device.memoryAllocated(), 10 * sizeof(float));
  }
  ASSERT_SAME_SIZE(device.memoryAllocated(), 0);
}
