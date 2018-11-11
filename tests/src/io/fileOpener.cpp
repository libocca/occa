#include <occa/io.hpp>
#include <occa/tools/env.hpp>
#include <occa/tools/testing.hpp>

void testFileOpeners();

int main(const int argc, const char **argv) {
  occa::env::OCCA_CACHE_DIR = occa::io::dirname(__FILE__);

  testFileOpeners();

  return 0;
}

void testFileOpeners() {
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

  occa::io::occaFileOpener occaOpener;
  ASSERT_FALSE(occaOpener.handles(""));
  ASSERT_FALSE(occaOpener.handles("foo.okl"));
  ASSERT_TRUE(occaOpener.handles("occa://"));
  ASSERT_TRUE(occaOpener.handles("occa://foo.okl"));

  ASSERT_EQ(occaOpener.expand("occa://"),
            occa::io::cachePath());
  ASSERT_EQ(occaOpener.expand("occa://foo.okl"),
            occa::io::libraryPath() + "foo.okl");
}
