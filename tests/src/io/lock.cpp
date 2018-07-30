/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 David Medina and Tim Warburton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <occa/io.hpp>
#include <occa/tools/env.hpp>
#include <occa/tools/testing.hpp>

void testInit();
void testAutoRelease();
void testStaleRelease();
void clearLocks();

int main(const int argc, const char **argv) {
  occa::env::OCCA_CACHE_DIR = occa::io::dirname(__FILE__);
  occa::settings()["locks/stale-warning"] = 0;
  occa::settings()["locks/stale-age"] = 0.2;

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
            + hash.toString()
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
