#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <occa/internal/io.hpp>
#include <occa/internal/utils/env.hpp>
#include <occa/internal/utils/testing.hpp>

void testInit();
void testAutoRelease();
void testStaleRelease();
void clearLocks();

int main(const int argc, const char **argv) {
#ifndef USE_CMAKE
  occa::env::OCCA_CACHE_DIR = occa::io::dirname(__FILE__);
#endif
  occa::settings()["locks/stale_warning"] = 0;
  occa::settings()["locks/stale_age"] = 0.2;

  srand(time(NULL));

  clearLocks();

  testInit();
  testAutoRelease();
  testStaleRelease();

  clearLocks();

  return 0;
}

void testInit() {
  occa::io::lock_t lock;
  ASSERT_FALSE(lock.isInitialized());
}

void testAutoRelease() {
  occa::hash_t hash = occa::hash(occa::toString(rand()));

  occa::io::lock_t lock1(hash, "tag");
  ASSERT_TRUE(lock1.isInitialized());
  ASSERT_EQ(lock1.dir(),
            occa::env::OCCA_CACHE_DIR
            + "locks/"
            + hash.getString()
            + "_tag");
  ASSERT_TRUE(lock1.isMine());

  ASSERT_TRUE(occa::io::isDir(lock1.dir()));

  occa::io::lock_t lock2(hash, "tag");
  ASSERT_TRUE(lock2.isMine());

  ASSERT_TRUE(occa::io::isDir(lock2.dir()));
  lock2.release();
  ASSERT_FALSE(occa::io::isDir(lock2.dir()));
}

void testStaleRelease() {
  occa::hash_t hash = occa::hash(occa::toString(rand()));

  occa::io::lock_t lock1(hash, "tag");
  ASSERT_TRUE(lock1.isMine());
  // Test cached isMine()
  ASSERT_TRUE(lock1.isMine());

  occa::io::lock_t lock2(hash, "tag");
  ASSERT_TRUE(lock2.isMine());

  // Wait 0.5 seconds until both locks are considered stale
  ::usleep(500000);

  // Kill the stale lock
  occa::io::lock_t lock3(hash, "tag");
  ASSERT_TRUE(lock3.isMine());

  lock1.release();
  lock2.release();
  lock3.release();

  ASSERT_FALSE(occa::io::isDir(lock1.dir()));
}

void clearLocks() {
  occa::sys::rmdir(occa::env::OCCA_CACHE_DIR + "locks",
                   true);
}
