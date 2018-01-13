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
#include "occa/defines.hpp"
#include "occa/tools/sys.hpp"
#include "occa/tools/gc.hpp"

class test : public occa::withRef {
public:
  static int count;

  test(const int idx) {}

  test(const test &other) :
    occa::withRef(other) {}

  test& operator = (const test &other) {
    changeRef(other);
    return *this;
  }

  ~test() {
    removeRef();
  }

  virtual void destructor() {
    --count;
  }
};

int test::count = 2;

int main(const int argc, const char **argv) {
  {
    test a1(1), a2(a1), a3(a1), a4(a1), a5(a1), a6(a4);
    test b1(2), b2(b1), b3(b1), b4(b1), b5(b1), b6(b4);

    a1 = b5;
    a2 = b2;
    a3 = b2;
    a4 = b3;
    a5 = b3;

    b1 = a5;
    b2 = a4;
    b3 = a3;
    b4 = a2;
    b5 = a1;
    OCCA_ERROR("Oh oh... some died: " << (2 - test::count),
               test::count == 2);
  }

  OCCA_ERROR("Oh oh... left alive: " << test::count,
             test::count == 0);
}
