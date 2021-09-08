#include <stdlib.h>
#include <time.h>

#include <occa/internal/io.hpp>
#include <occa/internal/utils/env.hpp>
#include <occa/internal/utils/testing.hpp>

void testCacheInfoMethods();
void testHashDir();
void testBuild();

int main(const int argc, const char **argv) {
#ifndef USE_CMAKE
  occa::env::OCCA_CACHE_DIR = occa::io::dirname(__FILE__);
#endif
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
}

void testHashDir() {
  occa::hash_t hash = occa::hash(occa::toString(rand()));
  const std::string cacheDir = (
    occa::io::cachePath() + hash.getString() + "/"
  );
  const std::string manualCacheDir = (
    occa::io::cachePath() + "1234/"
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
}

void testBuild() {
  // Build props
  occa::json props;
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
  occa::io::writeBuildFile("build.json", props);

  ASSERT_TRUE(occa::io::isFile("build.json"));

  occa::sys::rmrf("build.json");
}
