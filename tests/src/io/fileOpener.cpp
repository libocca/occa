#include <occa/io.hpp>
#include <occa/tools/env.hpp>
#include <occa/tools/testing.hpp>

void testDefaultFileOpener();
void testOccaFileOpener();

int main(const int argc, const char **argv) {
#ifndef USE_CMAKE
  occa::env::OCCA_CACHE_DIR = occa::io::dirname(__FILE__);
#endif

  testDefaultFileOpener();
  testOccaFileOpener();

  return 0;
}

void testDefaultFileOpener() {
  occa::io::defaultFileOpener defaultOpener;
  ASSERT_TRUE(defaultOpener.handles(""));
  ASSERT_TRUE(defaultOpener.handles("foo.okl"));
  ASSERT_TRUE(defaultOpener.handles("occa://foo.okl"));

  ASSERT_EQ(defaultOpener.expand(""),
            "");
  ASSERT_EQ(defaultOpener.expand("foo.okl"),
            "foo.okl");
  ASSERT_EQ(defaultOpener.expand("occa://foo.okl"),
            "occa://foo.okl");
}

void testOccaFileOpener() {
  occa::io::occaFileOpener occaOpener;
  ASSERT_FALSE(occaOpener.handles(""));
  ASSERT_FALSE(occaOpener.handles("foo.okl"));
  ASSERT_TRUE(occaOpener.handles("occa://"));
  ASSERT_TRUE(occaOpener.handles("occa://foo.okl"));

  ASSERT_EQ(occaOpener.expand("occa://"),
            "");
  ASSERT_EQ(occaOpener.expand("occa://foo.okl"),
            "");

  ASSERT_EQ(occaOpener.expand("occa://my_library/"),
            "");
  ASSERT_EQ(occaOpener.expand("occa://my_library/foo.okl"),
            "");

  occa::io::addLibraryPath("my_library", "foo");

  ASSERT_EQ(occaOpener.expand("occa://my_library/"),
            "foo/");
  ASSERT_EQ(occaOpener.expand("occa://my_library/foo.okl"),
            "foo/foo.okl");

  occa::io::addLibraryPath("my_library", "foo/");

  ASSERT_EQ(occaOpener.expand("occa://my_library/"),
            "foo/");
  ASSERT_EQ(occaOpener.expand("occa://my_library/foo.okl"),
            "foo/foo.okl");

  ASSERT_THROW(
    occa::io::addLibraryPath("", "foobar");
  );
  ASSERT_THROW(
    occa::io::addLibraryPath("a/b/c", "foo/");
  );
}
