#include <occa.hpp>
#include <occa/internal/utils/testing.hpp>

void testMalloc();
void testCopy();
void testPartialCopy();
void testSlice();
void testUnwrap();
void testCast();

int main(const int argc, const char **argv) {
  testMalloc();
  testCopy();
  testPartialCopy();
  testSlice();
  testUnwrap();
  testCast();

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

void testCast() {
  occa::device occa_device({{"mode", "Serial"}});

  occa::memory occa_memory = occa_device.malloc<double>(10);

  ASSERT_TRUE(occa::dtype::double_ == occa_memory.dtype());

  occa::memory casted_memory = occa_memory.cast(occa::dtype::byte);

  ASSERT_TRUE(occa::dtype::double_ == occa_memory.dtype());
  ASSERT_TRUE(occa::dtype::byte == casted_memory.dtype());

  ASSERT_EQ(occa_memory.byte_size(), casted_memory.byte_size());
}

void testCopy() {
  occa::device occa_device({{"mode", "Serial"}});

  const std::size_t N = 1024;

  std::vector<int> x_host(N,1);
  std::vector<int> y1_host(N,0);
  std::vector<int> y2_host(N,0);

  occa::memory x_device = occa_device.malloc<int>(N);
  occa::memory y_device = occa_device.malloc<int>(N);

  x_device.copyFrom(x_host.data());
  x_device.copyTo(y1_host.data());
  for (const auto& y : y1_host) {ASSERT_EQ(y,1);}

  y_device.copyFrom(x_device);
  y_device.copyTo(y2_host.data());
  for (const auto& y : y2_host) {ASSERT_EQ(y,1);}
}

void testPartialCopy() {
  occa::device occa_device({{"mode", "Serial"}});

  const std::size_t N = 1024 * 2 * 3 * 4;

  std::vector<int> x_host(N);
  std::vector<int> y_host(N);

  occa::memory x_device = occa_device.malloc<int>(N);
  occa::memory y_device = occa_device.malloc<int>(N);

  for (std::size_t n = 0; n < N; ++n) {
    x_host[n] = static_cast<int>(n);
  }
 
  std::size_t entries;
  std::size_t offset;

  //---[Host to device]-------
  // Last quarter
  entries = N/4;
  offset = 3*N/4;
  x_device.copyFrom(x_host.data() + offset, entries, offset);

  // First quarter
  offset = 0;
  x_device.copyFrom(x_host.data() + offset, entries, offset);

  // Middle-half
  entries = N/2;
  offset = N/4;
  x_device.copyFrom(x_host.data() + offset, entries, offset);

  //---[Device to device]-------
  // Middle third
  entries = N/3;
  offset = N/3;
  y_device.copyFrom(x_device, entries, offset, offset);

  // First third
  offset = 0;
  x_device.copyTo(y_device, entries, offset, offset);

  // Last third
  offset = 2 * N / 3;
  y_device.copyFrom(x_device, entries, offset, offset);

  //---[Device to host]-------
  // Last half
  entries = N/2;
  offset = N/2;
  y_device.copyTo(y_host.data() + offset, entries, offset);

  // First half
  offset = 0;
  y_device.copyTo(y_host.data() + offset, entries, offset);

  for (std::size_t n = 0; n < N; ++n) {
    ASSERT_EQ(x_host[n], y_host[n]);
  }
}
