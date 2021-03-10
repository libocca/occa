#include <algorithm>
#include <stdlib.h>
#include <time.h>

#include <occa/internal/io.hpp>
#include <occa/internal/utils.hpp>
#include <occa/internal/utils/testing.hpp>

void testPathMethods();
void testDirMethods();
void testIOMethods();

int main(const int argc, const char **argv) {
#ifndef USE_CMAKE
  occa::env::OCCA_CACHE_DIR = "/occa/cache/dir/";
#endif
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

  ASSERT_EQ(occa::io::getRelativePath("./a"),
            "a");
  ASSERT_EQ(occa::io::getRelativePath("./a/b"),
            "a/b");
  ASSERT_EQ(occa::io::getRelativePath(".a/b"),
            ".a/b");
  ASSERT_EQ(occa::io::getRelativePath("../a/b"),
            "../a/b");

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
            occa::env::CWD);
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
}

void testDirMethods() {
  const std::string ioDir = occa::io::dirname(__FILE__);
  const std::string testDir = occa::io::dirname(ioDir);

  // Find files
  occa::strVector files = occa::io::files(ioDir);
  ASSERT_EQ((int) files.size(),
            3);
  ASSERT_IN(ioDir + "cache.cpp", files);
  ASSERT_IN(ioDir + "lock.cpp", files);
  ASSERT_IN(ioDir + "utils.cpp", files);

  // Check if files exists
  ASSERT_TRUE(occa::io::exists(ioDir + "cache.cpp"));
  ASSERT_TRUE(occa::io::exists(ioDir + "lock.cpp"));
  ASSERT_TRUE(occa::io::exists(ioDir + "utils.cpp"));
  ASSERT_FALSE(occa::io::exists(ioDir + "foo.okl"));

  // Find directories
  occa::strVector dirs = occa::io::directories(ioDir);
  ASSERT_EQ((int) dirs.size(),
            0);
  dirs = occa::io::directories(testDir);
  ASSERT_EQ((int) dirs.size(),
            4);

  ASSERT_IN(testDir + "bin/", dirs);
  ASSERT_IN(testDir + "io/", dirs);
  ASSERT_IN(testDir + "lang/", dirs);
  ASSERT_IN(testDir + "utils/", dirs);
}

void testIOMethods() {
  const std::string test_foo = occa::env::CWD + "test_foo";

  std::string content = "start";
  for (int i = 0; i < 100; ++i) {
    content += occa::toString(rand());
  }
  occa::io::write(test_foo, content);
  ASSERT_TRUE(occa::io::exists(test_foo));

  // Read
  ASSERT_EQ(occa::io::read(test_foo),
            content);
  ASSERT_EQ(occa::io::read(test_foo, occa::enums::FILE_TYPE_BINARY),
            content);
  ASSERT_EQ(occa::io::read(test_foo, occa::enums::FILE_TYPE_PSEUDO),
            content);

  // C Read
  size_t charCount = occa::UDIM_DEFAULT;
  char *c = occa::io::c_read(test_foo, &charCount);
  ASSERT_EQ(charCount,
            content.size());
  delete [] c;

  occa::sys::rmrf(test_foo);
}
