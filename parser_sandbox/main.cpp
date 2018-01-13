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
#include <iostream>

#if 0
#include "occa.hpp"
#include "occa/tools/io.hpp"
#include "occa/parser/tools.hpp"

#include "tools.hpp"
#include "preprocessor.hpp"
#endif

#include "occa/tools/sys.hpp"

#include "basicParser.hpp"
#include "occa/parser/primitive.hpp"

int main(int argc, char **argv) {
#if 0
  std::string content = occa::io::read("cleanTest.c");
  const char *c = content.c_str();
  std::string processedContent;

  preprocessor_t preprocessor;
  preprocessor.process(c, processedContent);

  std::cout << processedContent << '\n';
#endif
  return 0;
}
