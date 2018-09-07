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
#include <algorithm>
#include <stdlib.h>
#include <time.h>

#include <occa/io.hpp>
#include <occa/tools/env.hpp>
#include <occa/tools/sys.hpp>
#include <occa/tools/testing.hpp>

void testPathMethods();
void testDirMethods();
void testIOMethods();

int main(const int argc, const char **argv) {
  occa::env::OCCA_CACHE_DIR = "/occa/cache/dir/";
  srand(time(NULL));

  testPathMethods();
  testDirMethods();
  testIOMethods();

  return 0;
}

void testPathMethods() {
  ASSERT_EQ(occa::io::cachePath(),
            occa::env::OCCA_CACHE_DIR + "cache/");

  ASSERT_EQ(occa::io::libraryPath(),
            occa::env::OCCA_CACHE_DIR + "libraries/");

  ASSERT_EQ(occa::io::endWithSlash("a"),
            "a/");
  ASSERT_EQ(occa::io::endWithSlash("a/"),
            "a/");

  ASSERT_EQ(occa::io::removeEndSlash("a"),
            "a");
  ASSERT_EQ(occa::io::removeEndSlash("a/"),
            "a");

  ASSERT_EQ(occa::io::slashToSnake(""),
            "");
  ASSERT_EQ(occa::io::slashToSnake("a b"),
            "a b");
  ASSERT_EQ(occa::io::slashToSnake("a/b/c"),
            "a_b_c");
  ASSERT_EQ(occa::io::slashToSnake("a//b//c"),
            "a__b__c");

  ASSERT_TRUE(occa::io::isAbsolutePath("/a"));
  ASSERT_FALSE(occa::io::isAbsolutePath(""));
  ASSERT_FALSE(occa::io::isAbsolutePath("a"));

  ASSERT_EQ(occa::io::convertSlashes("a"),
            "a");
  ASSERT_EQ(occa::io::convertSlashes("/a/b"),
            "/a/b");

  ASSERT_EQ(occa::io::expandEnvVariables("~"),
            occa::env::HOME);
  ASSERT_EQ(occa::io::expandEnvVariables("~/a"),
            occa::env::HOME + "a");
  ASSERT_EQ(occa::io::expandEnvVariables("~a"),
            "~a");
  ASSERT_EQ(occa::io::endWithSlash(
              occa::io::expandEnvVariables("$HOME")
            ),
            occa::env::HOME);
  ASSERT_EQ(occa::io::endWithSlash(
              occa::io::expandEnvVariables("${HOME}")
            ),
            occa::env::HOME);

  ASSERT_EQ(occa::io::binaryName("a"),
            "a");
  ASSERT_EQ(occa::io::binaryName("/a/b"),
            "/a/b");

  ASSERT_EQ(occa::io::basename("a.okl"),
            "a.okl");
  ASSERT_EQ(occa::io::basename("a.okl", false),
            "a");
  ASSERT_EQ(occa::io::basename("/a/b.okl"),
            "b.okl");
  ASSERT_EQ(occa::io::basename("/a/b.okl", false),
            "b");

  ASSERT_EQ(occa::io::dirname("b.okl"),
            occa::env::PWD);
  ASSERT_EQ(occa::io::dirname("/a/b.okl"),
            "/a/");
  ASSERT_EQ(occa::io::dirname("/a"),
            "/");
  ASSERT_EQ(occa::io::dirname("/a/"),
            "/");

  ASSERT_EQ(occa::io::extension("a"),
            "");
  ASSERT_EQ(occa::io::extension("a.okl"),
            "okl");
  ASSERT_EQ(occa::io::extension("a.test.okl"),
            "okl");

  ASSERT_EQ(occa::io::shortname("foo.okl"),
            "foo.okl");
  ASSERT_EQ(occa::io::shortname(occa::io::cachePath() + "hash/foo.okl"),
            "hash/foo.okl");
  ASSERT_EQ(occa::io::shortname(occa::io::libraryPath() + "lib/foo.okl"),
            "occa://lib/foo.okl");
}

void testDirMethods() {
  const std::string ioDir = occa::io::dirname(__FILE__);
  const std::string testDir = occa::io::dirname(ioDir);

  // Find files
  occa::strVector files = occa::io::files(ioDir);
  ASSERT_EQ((int) files.size(),
            4);
  ASSERT_IN(ioDir + "cache.cpp", files);
  ASSERT_IN(ioDir + "fileOpener.cpp", files);
  ASSERT_IN(ioDir + "lock.cpp", files);
  ASSERT_IN(ioDir + "utils.cpp", files);

  // Check if files exists
  ASSERT_TRUE(occa::io::exists(ioDir + "cache.cpp"));
  ASSERT_TRUE(occa::io::exists(ioDir + "fileOpener.cpp"));
  ASSERT_TRUE(occa::io::exists(ioDir + "lock.cpp"));
  ASSERT_TRUE(occa::io::exists(ioDir + "utils.cpp"));
  ASSERT_FALSE(occa::io::exists(ioDir + "foo.okl"));

  // Find directories
  occa::strVector dirs = occa::io::directories(ioDir);
  ASSERT_EQ((int) dirs.size(),
            0);
  dirs = occa::io::directories(testDir);
  ASSERT_EQ((int) dirs.size(),
            6);

  ASSERT_IN(testDir + "c/", dirs);
  ASSERT_IN(testDir + "io/", dirs);
  ASSERT_IN(testDir + "lang/", dirs);
  ASSERT_IN(testDir + "tools/", dirs);
}

void testIOMethods() {
  const std::string test_foo = occa::env::PWD + "test_foo";

  std::string content = "start";
  for (int i = 0; i < 100; ++i) {
    content += occa::toString(rand());
  }
  occa::io::write(test_foo, content);
  ASSERT_TRUE(occa::io::exists(test_foo));

  // Read
  ASSERT_EQ(occa::io::read(test_foo),
            content);
  ASSERT_EQ(occa::io::read(test_foo, true),
            content);

  // C Read
  size_t charCount = -1;
  char *c = occa::io::c_read(test_foo, &charCount);
  ASSERT_EQ(charCount,
            content.size());
  delete [] c;

  occa::sys::rmrf(test_foo);
}
