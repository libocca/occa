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
#include "occa/tools/testing.hpp"

#include "file.hpp"

int main(const int argc, const char **argv) {
  occa::lang::fileOrigin orig, orig2;
  std::string line = "#define foo bar";
  const char *c_line = line.c_str();
  const int foo = 8;
  const int bar = 12;
  orig.position = occa::lang::filePosition(10,
                                           c_line,
                                           c_line + foo);
  orig.push(true,
            NULL,
            occa::lang::filePosition(20,
                                     c_line,
                                     c_line + bar));
  orig2 = orig;
  orig.push(false,
            NULL,
            occa::lang::filePosition(30,
                                     c_line,
                                     c_line + foo));
  orig2.push(false,
             NULL,
             occa::lang::filePosition(40,
                                      c_line,
                                      c_line + bar));
  occa::lang::printer pout(std::cout);
  orig.print(pout);
  pout << "Test message\n\n";
  orig2.print(pout);
  pout << "Test message 2\n";
  return 0;
}
