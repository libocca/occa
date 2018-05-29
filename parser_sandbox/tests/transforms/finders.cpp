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

#include "../parserUtils.hpp"
#include "builtins/transforms/finders.hpp"

int main(const int argc, const char **argv) {
  parser.addAttribute<dummy>();
  parser.addAttribute<attributes::kernel>();
  parser.addAttribute<attributes::outer>();
  parser.addAttribute<attributes::inner>();
  parser.addAttribute<attributes::shared>();
  parser.addAttribute<attributes::exclusive>();

  statementPtrVector statements;
  parseSource(
    "@dummy int d1;\n"
    "@dummy void f1();\n"
    "@dummy void f2();\n"
    "@dummy void f3();\n"
    "@dummy void fd1() {}\n"
    "@dummy void fd2() {\n"
    "  @dummy int d2;\n"
    "  @dummy e1 + 1;\n"
    "  @dummy e2 += 1;\n"
    "  @dummy {\n"
    "    @dummy {}\n"
    "    @dummy int d3;\n"
    "    @dummy int d4;\n"
    "    @dummy e3++;\n"
    "    @dummy e4--;\n"
    "    @dummy e5 *= 1;\n"
    "  }\n"
    "}\n"
  );

  transforms::findStatements(statementType::block,
                             "dummy",
                             parser.root,
                             statements);
  OCCA_ASSERT_EQUAL(2,
                    (int) statements.size());
  statements.clear();

  transforms::findStatements(statementType::expression,
                             "dummy",
                             parser.root,
                             statements);
  OCCA_ASSERT_EQUAL(5,
                    (int) statements.size());
  statements.clear();

  transforms::findStatements(statementType::declaration,
                             "dummy",
                             parser.root,
                             statements);
  OCCA_ASSERT_EQUAL(4,
                    (int) statements.size());
  statements.clear();

  transforms::findStatements(statementType::function,
                             "dummy",
                             parser.root,
                             statements);
  OCCA_ASSERT_EQUAL(3,
                    (int) statements.size());
  statements.clear();

  transforms::findStatements(statementType::functionDecl,
                             "dummy",
                             parser.root,
                             statements);
  OCCA_ASSERT_EQUAL(2,
                    (int) statements.size());
  statements.clear();

  return 0;
}
