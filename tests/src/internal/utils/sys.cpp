#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sstream>

#include <occa.hpp>

#include <occa/internal/io.hpp>
#include <occa/internal/utils/testing.hpp>

void testRmrf();

int main(const int argc, const char **argv) {
  srand(time(NULL));

  testRmrf();

  return 0;
}

void testRmrf() {
  // Need to make sure "occa" is in the filepath
  occa::io::addLibraryPath("my_library", "/path/to/my_library/occa");

  ASSERT_TRUE(occa::sys::isSafeToRmrf("/a/occa/b"));
  ASSERT_TRUE(occa::sys::isSafeToRmrf("/a/b/../b/.occa"));
  ASSERT_TRUE(occa::sys::isSafeToRmrf("/a/occa_dir/b"));
  ASSERT_TRUE(occa::sys::isSafeToRmrf("/a/b/../b/.occa_dir"));
  ASSERT_TRUE(occa::sys::isSafeToRmrf("/../../../../../occa/a/b"));
  ASSERT_TRUE(occa::sys::isSafeToRmrf("~/c/.cache/occa"));
  ASSERT_TRUE(occa::sys::isSafeToRmrf("occa://my_library/"));
  ASSERT_TRUE(occa::sys::isSafeToRmrf("occa://my_library/file"));

  ASSERT_FALSE(occa::sys::isSafeToRmrf("/occa/a/.."));
  ASSERT_FALSE(occa::sys::isSafeToRmrf("/a/b/c/.."));
  ASSERT_FALSE(occa::sys::isSafeToRmrf("/a/b/c/../"));
  ASSERT_FALSE(occa::sys::isSafeToRmrf("/../../../../../a/b"));
  ASSERT_FALSE(occa::sys::isSafeToRmrf("~/c/.."));

  // Needs to have occa in the name
  const std::string dummyDir = "occa_foo_bar_test";

  occa::sys::rmrf(dummyDir);
  ASSERT_FALSE(occa::io::isDir(dummyDir));

  occa::sys::mkpath(dummyDir);
  ASSERT_TRUE(occa::io::isDir(dummyDir));

  occa::sys::rmrf(dummyDir);
  ASSERT_FALSE(occa::io::isDir(dummyDir));

  // Test safety on rmrf
  // Make 100% sure we're trying to delete the non-existent directory
  std::string filename = "/fake_occa_directory_for_this_sys_test";
  std::string expFilename = (
    occa::io::expandFilename("/fake_occa_directory_for_this_sys_test")
  );
  ASSERT_EQ(filename, expFilename);
  ASSERT_FALSE(occa::io::isDir(filename));
  ASSERT_FALSE(occa::io::isFile(filename));

  ASSERT_THROW(
    occa::sys::rmrf(filename);
  );

  occa::settings()["sys/safe_rmrf"] = false;
  occa::sys::rmrf(filename);
}
