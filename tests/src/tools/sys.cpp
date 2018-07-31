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
#include <sstream>

#include <occa.hpp>
#include <occa/tools/testing.hpp>

void testRmrf();

int main(const int argc, const char **argv) {
  srand(time(NULL));

  testRmrf();

  return 0;
}

void testRmrf() {
  ASSERT_TRUE(occa::sys::isSafeToRmrf("/a/b/c"));
  ASSERT_TRUE(occa::sys::isSafeToRmrf("/a/b/../b/c"));
  ASSERT_TRUE(occa::sys::isSafeToRmrf("/../../../../../a/b/c"));
  ASSERT_TRUE(occa::sys::isSafeToRmrf("~/c"));
  ASSERT_TRUE(occa::sys::isSafeToRmrf("occa://lib"));

  ASSERT_FALSE(occa::sys::isSafeToRmrf("/a/b/c/.."));
  ASSERT_FALSE(occa::sys::isSafeToRmrf("/a/b/c/../"));
  ASSERT_FALSE(occa::sys::isSafeToRmrf("/../../../../../a/b"));
  ASSERT_FALSE(occa::sys::isSafeToRmrf("~/c/.."));

  const std::string dummyDir = "occa_foo_bar_test";

  occa::sys::rmrf(dummyDir);
  ASSERT_FALSE(occa::io::isDir(dummyDir));

  occa::sys::mkdir(dummyDir);
  ASSERT_TRUE(occa::io::isDir(dummyDir));

  occa::sys::rmrf(dummyDir);
  ASSERT_FALSE(occa::io::isDir(dummyDir));

  // Test safety on rmrf
  // Make 100% sure we're trying to delete the non-existent directory
  std::string filename = "/fake_occa_directory_for_this_sys_test";
  std::string expFilename = (
    occa::io::filename("/fake_occa_directory_for_this_sys_test")
  );
  ASSERT_EQ(filename, expFilename);
  ASSERT_FALSE(occa::io::isDir(filename));
  ASSERT_FALSE(occa::io::isFile(filename));

  ASSERT_THROW(
    occa::sys::rmrf(filename);
  );

  occa::settings()["options/safe-rmrf"] = false;
  occa::sys::rmrf(filename);
}
