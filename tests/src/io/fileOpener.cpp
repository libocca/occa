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
