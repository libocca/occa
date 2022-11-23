#include <occa.hpp>
#include <occa/internal/utils/testing.hpp>

void testMalloc();
void testSlice();
void testUnwrap();

int main(const int argc, const char **argv) {
  testMalloc();
  testSlice();
  testUnwrap();

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

  mem = occa::malloc(bytes, hostPtr, {{"use_host_pointer", true}});
  ASSERT_EQ(mem.ptr<int>()[0], value);
  ASSERT_EQ(mem.ptr<int>(), hostPtr);

  occa::setDevice({
    {"mode", "Serial"},
    {"memory", {
      {"use_host_pointer", true}
    }}
  });

  mem = occa::malloc(bytes, hostPtr);
  ASSERT_EQ(((int*) mem.ptr())[0], value);
  ASSERT_EQ(mem.ptr<int>(), hostPtr);

  mem = occa::malloc(bytes, hostPtr, {{"use_host_pointer", false}});
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

  occa::device device({
    {"mode", "Serial"}
  });
  ASSERT_SAME_SIZE(device.memoryAllocated(), 0);
  ASSERT_SAME_SIZE(device.maxMemoryAllocated(), 0);

  {
    occa::memory mem = device.malloc<float>(10, data);
    ASSERT_SAME_SIZE(device.memoryAllocated(), 10 * sizeof(float));
    ASSERT_SAME_SIZE(device.maxMemoryAllocated(), 10 * sizeof(float));
    {
      occa::memory half1 = mem.slice(0, 5);
      occa::memory half2 = mem.slice(5);
    }

    ASSERT_SAME_SIZE(device.memoryAllocated(), 10 * sizeof(float));
    ASSERT_SAME_SIZE(device.maxMemoryAllocated(), 10 * sizeof(float));
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
    ASSERT_SAME_SIZE(device.maxMemoryAllocated(), 10 * sizeof(float));
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
    ASSERT_SAME_SIZE(device.maxMemoryAllocated(), 10 * sizeof(float));
  }
  ASSERT_SAME_SIZE(device.memoryAllocated(), 0);
  ASSERT_SAME_SIZE(device.maxMemoryAllocated(), 10 * sizeof(float));


  {
    occa::memory half1;
    occa::memory half2;
    {
      occa::memory mem = device.malloc<float>(10, data);
      half1 = mem.slice(0, 5);
      half2 = mem.slice(5);
      ASSERT_SAME_SIZE(device.memoryAllocated(), 10 * sizeof(float));
      ASSERT_SAME_SIZE(device.maxMemoryAllocated(), 10 * sizeof(float));
    }
    ASSERT_SAME_SIZE(device.memoryAllocated(), 10 * sizeof(float));
    ASSERT_SAME_SIZE(device.maxMemoryAllocated(), 10 * sizeof(float));
  }
  ASSERT_SAME_SIZE(device.memoryAllocated(), 0);
  ASSERT_SAME_SIZE(device.maxMemoryAllocated(), 10 * sizeof(float));
}

void testUnwrap() {
  occa::device occa_device({
    {"mode", "Serial"}
  });

  occa::memory occa_memory; 

   // Unwrapping uninitialized memory is undefined
  ASSERT_THROW(occa::unwrap(occa_memory););

  int* host_memory = new int[10];
  for (int i = 0; i < 10; ++i) host_memory[i] = 1;

  occa_memory = occa_device.malloc<int>(10,host_memory);

  ASSERT_EQ("Serial",occa_memory.mode());
  ASSERT_TRUE(occa_memory.getDevice() == occa_device);

  int* mode_memory = *static_cast<int**>(occa::unwrap(occa_memory));

  for (int i = 0; i < 10; ++i) {
    ASSERT_EQ(host_memory[i],mode_memory[i]);
  }

  delete[] host_memory;
}
