#define OCCA_DISABLE_VARIADIC_MACROS

#include <occa.hpp>
#include <occa.h>
#include <occa/c/experimental.h>
#include <occa/internal/c/types.hpp>
#include <occa/internal/utils/testing.hpp>

void testInit();
void testReserve();

int main(const int argc, const char **argv) {
  testInit();
  testReserve();

  return 0;
}

void testInit() {
  occaMemoryPool memPool = occaUndefined;
  occaJson props = (
    occaJsonParse("{foo: 'bar'}")
  );

  ASSERT_TRUE(occaIsUndefined(memPool));
  ASSERT_EQ(memPool.type,
            OCCA_UNDEFINED);
  ASSERT_FALSE(occaMemoryPoolIsInitialized(memPool));

  memPool = occaCreateMemoryPool(props);
  ASSERT_FALSE(occaIsUndefined(memPool));
  ASSERT_EQ(memPool.type,
            OCCA_MEMORYPOOL);
  ASSERT_TRUE(occaMemoryPoolIsInitialized(memPool));

  ASSERT_EQ(occa::c::device(occaMemoryPoolGetDevice(memPool)),
            occa::host());

  occaJson memPoolProps = occaMemoryPoolGetProperties(memPool);
  occaType memPoolMode = occaJsonObjectGet(memPoolProps, "foo", occaUndefined);
  ASSERT_EQ((const char*) occaJsonGetString(memPoolMode),
            (const char*) "bar");

  occaFree(&props);
  occaFree(&memPool);
}

void testReserve() {
  #define ASSERT_SAME_SIZE(a, b) \
  ASSERT_EQ((size_t) (a), (size_t) (b))

  float *data = new float[30];
  float *test = new float[30];
  for (int i = 0; i < 30; ++i) {
    data[i] = i;
  }

  occaMemoryPool memPool = occaCreateMemoryPool(occaDefault);

  /*Set aligment to 5*sizeof(float) bytes*/
  occaMemoryPoolSetAlignment(memPool, 5 * sizeof(float));

  /*Set a size for the memoryPool*/
  occaMemoryPoolResize(memPool, 10 * sizeof(float));

  occaDevice device = occaMemoryPoolGetDevice(memPool);

  ASSERT_SAME_SIZE(occaDeviceMemoryAllocated(device), 10 * sizeof(float));
  ASSERT_SAME_SIZE(occaMemoryPoolSize(memPool), 10 * sizeof(float));
  ASSERT_SAME_SIZE(occaMemoryPoolReserved(memPool), 0);

  /*Make a reservation*/
  occaMemory mem = occaMemoryPoolTypedReserve(memPool, 10, occaDtypeFloat);
  occaCopyPtrToMem(mem, data,
                   occaAllBytes, 0,
                   occaDefault);

  ASSERT_SAME_SIZE(occaDeviceMemoryAllocated(device), 10 * sizeof(float));
  ASSERT_SAME_SIZE(occaMemoryPoolSize(memPool), 10 * sizeof(float));
  ASSERT_SAME_SIZE(occaMemoryPoolReserved(memPool), 10 * sizeof(float));

  {
    /*Test slicing mem in memoryPool. Should not trigger reallocation or
      increase in memoryPool's reservation size*/
    occaMemory half1 = occaMemorySlice(mem, 0, 5);
    occaMemory half2 = occaMemorySlice(mem, 5, occaAllBytes);

    ASSERT_SAME_SIZE(occaDeviceMemoryAllocated(device), 10 * sizeof(float));
    ASSERT_SAME_SIZE(occaMemoryPoolSize(memPool), 10 * sizeof(float));
    ASSERT_SAME_SIZE(occaMemoryPoolReserved(memPool), 10 * sizeof(float));

    occaCopyMemToPtr(test, half1,
                     occaAllBytes, 0,
                     occaDefault);
    for (int i = 0; i < 5; ++i) {
      ASSERT_EQ(static_cast<int>(test[i]), i);
    }
    occaCopyMemToPtr(test, half2,
                     occaAllBytes, 0,
                     occaDefault);
    for (int i = 0; i < 5; ++i) {
      ASSERT_EQ(static_cast<int>(test[i]), i+5);
    }

    /*Trigger resize*/
    occaMemory mem2 = occaMemoryPoolReserve(memPool, 10 * sizeof(float));

    ASSERT_SAME_SIZE(occaDeviceMemoryAllocated(device), 20 * sizeof(float));
    ASSERT_SAME_SIZE(occaMemoryPoolSize(memPool), 20 * sizeof(float));
    ASSERT_SAME_SIZE(occaMemoryPoolReserved(memPool), 20 * sizeof(float));

    occaCopyMemToPtr(test, mem,
                     occaAllBytes, 0,
                     occaDefault);
    for (int i = 0; i < 10; ++i) {
      ASSERT_EQ(static_cast<int>(test[i]), i);
    }
    occaCopyMemToPtr(test, half1,
                     occaAllBytes, 0,
                     occaDefault);
    for (int i = 0; i < 5; ++i) {
      ASSERT_EQ(static_cast<int>(test[i]), i);
    }
    occaCopyMemToPtr(test, half2,
                     occaAllBytes, 0,
                     occaDefault);
    for (int i = 0; i < 5; ++i) {
      ASSERT_EQ(static_cast<int>(test[i]), i+5);
    }

    occaFree(&half1);
    occaFree(&half2);
    occaFree(&mem2);
  }

  /*Delete buffers, memoryPool size does not change, but reservation is smaller*/
  ASSERT_SAME_SIZE(occaDeviceMemoryAllocated(device), 20 * sizeof(float));
  ASSERT_SAME_SIZE(occaMemoryPoolSize(memPool), 20 * sizeof(float));
  ASSERT_SAME_SIZE(occaMemoryPoolReserved(memPool), 10 * sizeof(float));

  /*Reserve again, should not trigger a resize*/
  occaMemory mem2 = occaMemoryPoolReserve(memPool, 10 * sizeof(float));
  occaCopyPtrToMem(mem2, data+10,
                   occaAllBytes, 0,
                   occaDefault);

  ASSERT_SAME_SIZE(occaDeviceMemoryAllocated(device), 20 * sizeof(float));
  ASSERT_SAME_SIZE(occaMemoryPoolSize(memPool), 20 * sizeof(float));
  ASSERT_SAME_SIZE(occaMemoryPoolReserved(memPool), 20 * sizeof(float));

  /*Trigger resize*/
  occaMemory mem3 = occaMemoryPoolReserve(memPool, 5 * sizeof(float));
  occaMemory mem4 = occaMemoryPoolReserve(memPool, 5 * sizeof(float));
  occaCopyPtrToMem(mem3, data+20,
                   occaAllBytes, 0,
                   occaDefault);
  occaCopyPtrToMem(mem4, data+25,
                   occaAllBytes, 0,
                   occaDefault);

  ASSERT_SAME_SIZE(occaDeviceMemoryAllocated(device), 30 * sizeof(float));
  ASSERT_SAME_SIZE(occaMemoryPoolSize(memPool), 30 * sizeof(float));
  ASSERT_SAME_SIZE(occaMemoryPoolReserved(memPool), 30 * sizeof(float));

  occaCopyMemToPtr(test, mem2,
                   occaAllBytes, 0,
                   occaDefault);
  for (int i = 0; i < 10; ++i) {
    ASSERT_EQ(static_cast<int>(test[i]), i+10);
  }

  /*Delete mem and mem3 to make gaps*/
  occaFree(&mem);
  occaFree(&mem3);
  ASSERT_SAME_SIZE(occaDeviceMemoryAllocated(device), 30 * sizeof(float));
  ASSERT_SAME_SIZE(occaMemoryPoolSize(memPool), 30 * sizeof(float));
  ASSERT_SAME_SIZE(occaMemoryPoolReserved(memPool), 15 * sizeof(float));

  /*Trigger a resize again, which shifts mem2 and mem4 */
  mem = occaMemoryPoolReserve(memPool, 20 * sizeof(float));
  ASSERT_SAME_SIZE(occaDeviceMemoryAllocated(device), 35 * sizeof(float));
  ASSERT_SAME_SIZE(occaMemoryPoolSize(memPool), 35 * sizeof(float));
  ASSERT_SAME_SIZE(occaMemoryPoolReserved(memPool), 35 * sizeof(float));

  occaCopyMemToPtr(test, mem2,
                   occaAllBytes, 0,
                   occaDefault);
  for (int i = 0; i < 10; ++i) {
    ASSERT_EQ(static_cast<int>(test[i]), i+10);
  }
  occaCopyMemToPtr(test, mem4,
                   occaAllBytes, 0,
                   occaDefault);
  for (int i = 0; i < 5; ++i) {
    ASSERT_EQ(static_cast<int>(test[i]), i+25);
  }

  /*Manually free mem2*/
  occaFree(&mem2);
  ASSERT_SAME_SIZE(occaDeviceMemoryAllocated(device), 35 * sizeof(float));
  ASSERT_SAME_SIZE(occaMemoryPoolSize(memPool), 35 * sizeof(float));
  ASSERT_SAME_SIZE(occaMemoryPoolReserved(memPool), 25 * sizeof(float));

  /*Shrink pool to fit*/
  occaMemoryPoolShrinkToFit(memPool);
  ASSERT_SAME_SIZE(occaDeviceMemoryAllocated(device), 25 * sizeof(float));
  ASSERT_SAME_SIZE(occaMemoryPoolSize(memPool), 25 * sizeof(float));
  ASSERT_SAME_SIZE(occaMemoryPoolReserved(memPool), 25 * sizeof(float));

  occaCopyMemToPtr(test, mem4,
                   occaAllBytes, 0,
                   occaDefault);
  for (int i = 0; i < 5; ++i) {
    ASSERT_EQ(static_cast<int>(test[i]), i+25);
  }

  occaFree(&mem4);
  ASSERT_SAME_SIZE(occaDeviceMemoryAllocated(device), 25 * sizeof(float));
  ASSERT_SAME_SIZE(occaMemoryPoolSize(memPool), 25 * sizeof(float));
  ASSERT_SAME_SIZE(occaMemoryPoolReserved(memPool), 20 * sizeof(float));

  occaFree(&mem);
  ASSERT_SAME_SIZE(occaDeviceMemoryAllocated(device), 25 * sizeof(float));
  ASSERT_SAME_SIZE(occaMemoryPoolSize(memPool), 25 * sizeof(float));
  ASSERT_SAME_SIZE(occaMemoryPoolReserved(memPool), 0 * sizeof(float));

  occaFree(&memPool);
  ASSERT_SAME_SIZE(occaDeviceMemoryAllocated(device), 0);

  delete[] test;
  delete[] data;
}
