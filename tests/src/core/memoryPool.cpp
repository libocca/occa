#include <occa.hpp>
#include <occa/internal/utils/testing.hpp>

void testReserve();

int main(const int argc, const char **argv) {
  testReserve();

  return 0;
}

void testReserve() {
#define ASSERT_SAME_SIZE(a, b) \
  ASSERT_EQ((size_t) (a), (size_t) (b))

  float *data = new float[30];
  float *test = new float[30];
  for (int i = 0; i < 30; ++i) {
    data[i] = i;
  }

  occa::device device({
    {"mode", "Serial"}
  });

  occa::experimental::memoryPool memPool = device.createMemoryPool();

  /*Set aligment to 5*sizeof(float) bytes*/
  memPool.setAlignment(5 * sizeof(float));

  /*Set a size for the memoryPool*/
  memPool.resize(10 * sizeof(float));

  ASSERT_SAME_SIZE(device.memoryAllocated(), 10 * sizeof(float));
  ASSERT_SAME_SIZE(memPool.size(), 10 * sizeof(float));
  ASSERT_SAME_SIZE(memPool.reserved(), 0);

  /*Make a reservation*/
  occa::memory mem = memPool.reserve<float>(10);
  mem.copyFrom(data);

  ASSERT_SAME_SIZE(device.memoryAllocated(), 10 * sizeof(float));
  ASSERT_SAME_SIZE(memPool.size(), 10 * sizeof(float));
  ASSERT_SAME_SIZE(memPool.reserved(), 10 * sizeof(float));
  {
    /*Test slicing mem in memoryPool. Should not trigger reallocation or
      increase in memoryPool's reservation size*/
    occa::memory half1 = mem.slice(0, 5);
    occa::memory half2 = mem.slice(5);

    ASSERT_SAME_SIZE(device.memoryAllocated(), 10 * sizeof(float));
    ASSERT_SAME_SIZE(memPool.size(), 10 * sizeof(float));
    ASSERT_SAME_SIZE(memPool.reserved(), 10 * sizeof(float));

    half1.copyTo(test);
    for (int i = 0; i < 5; ++i) {
      ASSERT_EQ(static_cast<int>(test[i]), i);
    }
    half2.copyTo(test);
    for (int i = 0; i < 5; ++i) {
      ASSERT_EQ(static_cast<int>(test[i]), i+5);
    }

    /*Trigger resize*/
    occa::memory mem2 = memPool.reserve<float>(10);

    ASSERT_SAME_SIZE(device.memoryAllocated(), 20 * sizeof(float));
    ASSERT_SAME_SIZE(memPool.size(), 20 * sizeof(float));
    ASSERT_SAME_SIZE(memPool.reserved(), 20 * sizeof(float));

    mem.copyTo(test);
    for (int i = 0; i < 10; ++i) {
      ASSERT_EQ(static_cast<int>(test[i]), i);
    }
    half1.copyTo(test);
    for (int i = 0; i < 5; ++i) {
      ASSERT_EQ(static_cast<int>(test[i]), i);
    }
    half2.copyTo(test);
    for (int i = 0; i < 5; ++i) {
      ASSERT_EQ(static_cast<int>(test[i]), i+5);
    }
  }

  /*Delete buffers, memoryPool size does not change, but reservation is smaller*/
  ASSERT_SAME_SIZE(device.memoryAllocated(), 20 * sizeof(float));
  ASSERT_SAME_SIZE(memPool.size(), 20 * sizeof(float));
  ASSERT_SAME_SIZE(memPool.reserved(), 10 * sizeof(float));

  /*Reserve again, should not trigger a resize*/
  occa::memory mem2 = memPool.reserve<float>(10);
  mem2.copyFrom(data+10);

  ASSERT_SAME_SIZE(device.memoryAllocated(), 20 * sizeof(float));
  ASSERT_SAME_SIZE(memPool.size(), 20 * sizeof(float));
  ASSERT_SAME_SIZE(memPool.reserved(), 20 * sizeof(float));

  /*Trigger resize*/
  occa::memory mem3 = memPool.reserve<float>(5);
  occa::memory mem4 = memPool.reserve<float>(5);
  mem3.copyFrom(data+20);
  mem4.copyFrom(data+25);

  ASSERT_SAME_SIZE(device.memoryAllocated(), 30 * sizeof(float));
  ASSERT_SAME_SIZE(memPool.size(), 30 * sizeof(float));
  ASSERT_SAME_SIZE(memPool.reserved(), 30 * sizeof(float));

  mem2.copyTo(test);
  for (int i = 0; i < 10; ++i) {
    ASSERT_EQ(static_cast<int>(test[i]), i+10);
  }

  /*Delete mem and mem3 to make gaps*/
  mem.free();
  mem3.free();
  ASSERT_SAME_SIZE(device.memoryAllocated(), 30 * sizeof(float));
  ASSERT_SAME_SIZE(memPool.size(), 30 * sizeof(float));
  ASSERT_SAME_SIZE(memPool.reserved(), 15 * sizeof(float));

  /*Trigger a resize again, which shifts mem2 and mem4 */
  mem = memPool.reserve<float>(20);
  ASSERT_SAME_SIZE(device.memoryAllocated(), 35 * sizeof(float));
  ASSERT_SAME_SIZE(memPool.size(), 35 * sizeof(float));
  ASSERT_SAME_SIZE(memPool.reserved(), 35 * sizeof(float));

  mem2.copyTo(test);
  for (int i = 0; i < 10; ++i) {
    ASSERT_EQ(static_cast<int>(test[i]), i+10);
  }
  mem4.copyTo(test);
  for (int i = 0; i < 5; ++i) {
    ASSERT_EQ(static_cast<int>(test[i]), i+25);
  }

  /*Manually free mem2*/
  mem2.free();
  ASSERT_SAME_SIZE(device.memoryAllocated(), 35 * sizeof(float));
  ASSERT_SAME_SIZE(memPool.size(), 35 * sizeof(float));
  ASSERT_SAME_SIZE(memPool.reserved(), 25 * sizeof(float));

  /*Shrink pool to fit*/
  memPool.resize(memPool.reserved());
  ASSERT_SAME_SIZE(device.memoryAllocated(), 25 * sizeof(float));
  ASSERT_SAME_SIZE(memPool.size(), 25 * sizeof(float));
  ASSERT_SAME_SIZE(memPool.reserved(), 25 * sizeof(float));

  mem4.copyTo(test);
  for (int i = 0; i < 5; ++i) {
    ASSERT_EQ(static_cast<int>(test[i]), i+25);
  }

  mem4.free();
  ASSERT_SAME_SIZE(device.memoryAllocated(), 25 * sizeof(float));
  ASSERT_SAME_SIZE(memPool.size(), 25 * sizeof(float));
  ASSERT_SAME_SIZE(memPool.reserved(), 20 * sizeof(float));

  mem.free();
  ASSERT_SAME_SIZE(device.memoryAllocated(), 25 * sizeof(float));
  ASSERT_SAME_SIZE(memPool.size(), 25 * sizeof(float));
  ASSERT_SAME_SIZE(memPool.reserved(), 0 * sizeof(float));

  memPool.free();
  ASSERT_SAME_SIZE(device.memoryAllocated(), 0);

  delete[] test;
  delete[] data;
}
