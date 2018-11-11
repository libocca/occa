#include <occa/tools/testing.hpp>

#include <occa.hpp>

void testPtrRange();
void testUva();
void testUvaNull();

int main(const int argc, const char **argv) {
  testPtrRange();
  testUva();
  testUvaNull();

  return 0;
}

void testPtrRange() {
  occa::ptrRange range = occa::ptrRange();
  ASSERT_EQ(range.start,
            (char*) NULL);
  ASSERT_EQ(range.end,
            (char*) NULL);

  range = occa::ptrRange((void*) 10,
                         10);
  // [10,20) = [10,20)
  ASSERT_EQ(range,
            occa::ptrRange((void*) 10,
                           10));
  // [10,20) !n [5, 10)
  ASSERT_NEQ(range,
             occa::ptrRange((void*) 5,
                            5));
  // [10,20) !n [20, 30)
  ASSERT_NEQ(range,
             occa::ptrRange((void*) 20,
                            10));
  // [10,20) n [15,25)
  ASSERT_EQ(range,
             occa::ptrRange((void*) 15,
                            10));
  // [10,20) !n [0,5)
  ASSERT_NEQ(range,
             occa::ptrRange((void*) 0,
                            5));
  // [10,20) !n [25,35)
  ASSERT_NEQ(range,
             occa::ptrRange((void*) 25,
                            10));

  // [10,20) == [11,16)
  ASSERT_FALSE(range < occa::ptrRange((void*) 11,
                                      5));
  // [10,20) == [11,21)
  ASSERT_FALSE(range < occa::ptrRange((void*) 11,
                                      10));
  // [10,20) > [0,5)
  ASSERT_FALSE(range < occa::ptrRange((void*) 0,
                                      5));
  // [10,20) < [21,31)
  ASSERT_TRUE(range < occa::ptrRange((void*) 21,
                                     10));

  std::cout << "Testing ptrRange output: " << range << '\n';
}

void testUva() {
  size_t bytes = 10 * sizeof(int);
  int *ptr = (int*) occa::umalloc(bytes);

  occa::modeMemory_t *modeMemory = occa::uvaToMemory(ptr);

  ASSERT_NEQ(ptr,
             (int*) NULL);
  ASSERT_TRUE(occa::memory(modeMemory).isInitialized());

  // Managed
  ASSERT_TRUE(occa::isManaged(ptr));
  occa::stopManaging(ptr);
  ASSERT_FALSE(occa::isManaged(ptr));
  occa::startManaging(ptr);
  ASSERT_TRUE(occa::isManaged(ptr));

  // Sync
  ASSERT_FALSE(occa::needsSync(ptr));

  occa::syncToDevice(ptr);
  occa::syncToHost(ptr);

  occa::syncMemToDevice(modeMemory);
  occa::syncMemToHost(modeMemory);

  occa::sync(ptr);
  occa::dontSync(ptr);

  occa::removeFromStaleMap(modeMemory);
  occa::removeFromStaleMap(ptr);

  occa::freeUvaPtr(ptr);

  ASSERT_EQ(occa::uvaToMemory(ptr),
            (occa::modeMemory_t*) NULL);
}

void testUvaNull() {
  int *ptr = new int[2];

  ASSERT_EQ(occa::uvaToMemory(NULL),
            (occa::modeMemory_t*) NULL);
  ASSERT_EQ(occa::uvaToMemory(ptr),
            (occa::modeMemory_t*) NULL);

  occa::startManaging(NULL);
  occa::startManaging(ptr);

  occa::stopManaging(NULL);
  occa::stopManaging(ptr);

  occa::syncToDevice(NULL);
  occa::syncToDevice(ptr);

  occa::syncToHost(NULL);
  occa::syncToHost(ptr);

  occa::syncMemToDevice(NULL);
  occa::syncMemToHost(NULL);

  ASSERT_FALSE(occa::needsSync(NULL));
  ASSERT_FALSE(occa::needsSync(ptr));

  occa::sync(NULL);
  occa::sync(ptr);

  occa::dontSync(NULL);
  occa::dontSync(ptr);

  occa::freeUvaPtr(NULL);
  ASSERT_THROW(
    occa::freeUvaPtr(ptr);
  );

  delete [] ptr;
}
