#include <occa/internal/utils/testing.hpp>

#include <occa.hpp>

void testUva();
void testUvaNull();

int main(const int argc, const char **argv) {
  testUva();
  testUvaNull();

  return 0;
}

void testUva() {
  int *ptr = occa::umalloc<int>(10);

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
