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

#include <occa/io.hpp>
#include <occa/tools/env.hpp>
#include <occa/tools/testing.hpp>

void testCacheInfoMethods();
void testHashDir();
void testBuild();

int main(const int argc, const char **argv) {
  occa::env::OCCA_CACHE_DIR = occa::io::dirname(__FILE__);
  srand(time(NULL));

  testCacheInfoMethods();
  testHashDir();
  testBuild();

  occa::sys::rmdir(occa::env::OCCA_CACHE_DIR + "locks",
                   true);

  return 0;
}

void testCacheInfoMethods() {
  // isCached
  ASSERT_FALSE(occa::io::isCached(""));
  ASSERT_FALSE(occa::io::isCached("foo"));
  ASSERT_FALSE(occa::io::isCached("occa://foo.okl"));
  ASSERT_FALSE(occa::io::isCached("occa://lib/foo.okl"));
  ASSERT_FALSE(occa::io::isCached(
                occa::env::OCCA_CACHE_DIR + "foo.okl"
              ));
  ASSERT_TRUE(occa::io::isCached(
                occa::env::OCCA_CACHE_DIR + "cache/foo.okl"
              ));
  ASSERT_FALSE(occa::io::isCached(
                occa::env::OCCA_CACHE_DIR + "libraries/foo.okl"
              ));
  ASSERT_TRUE(occa::io::isCached(
                occa::env::OCCA_CACHE_DIR + "libraries/lib/cache/foo.okl"
              ));

  // getLibraryName
  ASSERT_EQ(occa::io::getLibraryName(""),
            "");
  ASSERT_EQ(occa::io::getLibraryName("foo.okl"),
            "");
  ASSERT_EQ(occa::io::getLibraryName("occa://lib/foo.okl"),
            "lib");
  ASSERT_EQ(occa::io::getLibraryName(
              occa::env::OCCA_CACHE_DIR + "libraries/lib/cache/foo.okl"
            ),
            "lib");
}

void testHashDir() {
  occa::hash_t hash = occa::hash(occa::toString(rand()));
  const std::string cacheDir = (
    occa::io::cachePath() + hash.toString() + "/"
  );
  const std::string cacheLibDir = (
    occa::io::libraryPath() + "lib/cache/" + hash.toString() + "/"
  );
  const std::string manualCacheDir = (
    occa::io::cachePath() + "1234/"
  );
  const std::string manualCacheLibDir = (
    occa::io::libraryPath() + "lib/cache/1234/"
  );

  // Default
  ASSERT_EQ(occa::io::hashDir(hash),
            cacheDir);
  ASSERT_EQ(occa::io::hashDir("", hash),
            cacheDir);

  // Non-cached files
  ASSERT_EQ(occa::io::hashDir("foo.okl", hash),
            cacheDir);
  ASSERT_EQ(occa::io::hashDir("dir/foo.okl", hash),
            cacheDir);
  ASSERT_EQ(occa::io::hashDir("occa://foo.okl", hash),
            cacheDir);
  ASSERT_EQ(occa::io::hashDir("occa://lib/foo.okl", hash),
            cacheDir);
  ASSERT_EQ(occa::io::hashDir("occa://lib/dir/foo.okl", hash),
            cacheDir);

  // Cached files
  ASSERT_EQ(occa::io::hashDir(manualCacheDir + "foo.okl", hash),
            manualCacheDir);
  ASSERT_EQ(occa::io::hashDir(manualCacheDir + "dir/foo.okl", hash),
            manualCacheDir);
  ASSERT_EQ(occa::io::hashDir(manualCacheLibDir + "foo.okl", hash),
            manualCacheLibDir);
  ASSERT_EQ(occa::io::hashDir(manualCacheLibDir + "dir/foo.okl", hash),
            manualCacheLibDir);
}

void testBuild() {
  // Build props
  occa::properties props;
  ASSERT_EQ(props.size(),
            0);
  occa::io::setBuildProps(props);
  ASSERT_EQ(props.size(),
            3);
  ASSERT_TRUE(props.has("date"));
  ASSERT_TRUE(props.has("human_date"));
  // Counts as 1
  ASSERT_TRUE(props.has("version/occa"));
  ASSERT_TRUE(props.has("version/okl"));

  // Write build file
  occa::hash_t hash = occa::hash(occa::toString(rand()));
  occa::io::writeBuildFile("build.json", hash, props);

  ASSERT_TRUE(occa::io::isFile("build.json"));

  occa::sys::rmrf("build.json");
}
