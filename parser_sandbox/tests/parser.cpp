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

#include "parser.hpp"

void testPeek();
void testParse();

using namespace occa::lang;

//---[ Util Methods ]-------------------
std::string source;
parser_t parser;

void setSource(const std::string &s) {
  source = s;
  parser.setSource(s, false);
}
//======================================

//---[ Macro Util Methods ]-------------
#define testStatementPeek(str_, type_)              \
  setSource(str_);                                  \
  OCCA_ASSERT_EQUAL_BINARY(type_,                   \
                           parser.peek());          \
  OCCA_ASSERT_TRUE(parser.success)

#define testStatementParse(str_, type_)             \
  setSource(str_);                                  \
  OCCA_ASSERT_EQUAL(1,                              \
                    parser.root.size());            \
  OCCA_ASSERT_EQUAL_BINARY(type_,                   \
                           parser.root[0]->type())  \
  OCCA_ASSERT_TRUE(parser.success)
//======================================

//---[ Tests ]--------------------------
int main(const int argc, const char **argv) {
  testPeek();
  // testParse();

  return 0;
}

void testPeek() {
  testStatementPeek("",
                    statementType::empty);

  testStatementPeek("#pragma occa test",
                    statementType::pragma);
  testStatementPeek("#pragma",
                    statementType::pragma);

  testStatementPeek("{}",
                    statementType::block);

  testStatementPeek("public:",
                    statementType::classAccess);
  testStatementPeek("protected:",
                    statementType::classAccess);
  testStatementPeek("private:",
                    statementType::classAccess);

  testStatementPeek("1 + 2;",
                    statementType::expression);
  testStatementPeek("\"a\";",
                    statementType::expression);
  testStatementPeek("'a';",
                    statementType::expression);
  testStatementPeek("sizeof 3;",
                    statementType::expression);

  testStatementPeek("int a = 0;",
                    statementType::declaration);
  testStatementPeek("const int a = 0;",
                    statementType::declaration);
  testStatementPeek("long long a, b = 0, *c = 0;",
                    statementType::declaration);

  testStatementPeek("goto foo;",
                    statementType::goto_);

  testStatementPeek("foo:",
                    statementType::gotoLabel);

  testStatementPeek("namespace foo {}",
                    statementType::namespace_);

  testStatementPeek("if (true) {}",
                    statementType::if_);
  testStatementPeek("else if (true) {}",
                    statementType::elif_);
  testStatementPeek("else {}",
                    statementType::else_);

  testStatementPeek("for () {}",
                    statementType::for_);

  testStatementPeek("while () {}",
                    statementType::while_);
  testStatementPeek("do {} while ()",
                    statementType::while_);

  testStatementPeek("switch () {}",
                    statementType::switch_);
  testStatementPeek("case foo:",
                    statementType::case_);
  testStatementPeek("continue;",
                    statementType::continue_);
  testStatementPeek("break;",
                    statementType::break_);

  testStatementPeek("return 0;",
                    statementType::return_);

  testStatementPeek("@attr",
                    statementType::attribute);
  testStatementPeek("@attr()",
                    statementType::attribute);
}

void testParse() {
  testStatementParse("",
                     statementType::empty);

  testStatementParse("#pragma occa test",
                     statementType::pragma);
  testStatementParse("#pragma",
                     statementType::pragma);

  testStatementParse("{}",
                     statementType::block);

  testStatementParse("public:",
                     statementType::classAccess);
  testStatementParse("protected:",
                     statementType::classAccess);
  testStatementParse("private:",
                     statementType::classAccess);

  testStatementParse("1 + 2;",
                     statementType::expression);
  testStatementParse("\"a\";",
                     statementType::expression);
  testStatementParse("'a';",
                     statementType::expression);
  testStatementParse("sizeof 3;",
                     statementType::expression);

  testStatementParse("int a = 0;",
                     statementType::declaration);
  testStatementParse("const int a = 0;",
                     statementType::declaration);
  testStatementParse("long long a, b = 0, *c = 0;",
                     statementType::declaration);

  testStatementParse("goto foo;",
                     statementType::goto_);

  testStatementParse("foo:",
                     statementType::gotoLabel);

  testStatementParse("namespace foo {}",
                     statementType::namespace_);

  testStatementParse("if (true) {}",
                     statementType::if_);
  testStatementParse("else if (true) {}",
                     statementType::elif_);
  testStatementParse("else {}",
                     statementType::else_);

  testStatementParse("for () {}",
                     statementType::for_);

  testStatementParse("while () {}",
                     statementType::while_);
  testStatementParse("do {} while ()",
                     statementType::while_);

  testStatementParse("switch () {}",
                     statementType::switch_);
  testStatementParse("case foo:",
                     statementType::case_);
  testStatementParse("continue;",
                     statementType::continue_);
  testStatementParse("break;",
                     statementType::break_);

  testStatementParse("return 0;",
                     statementType::return_);

  testStatementParse("@attr",
                     statementType::attribute);
  testStatementParse("@attr()",
                     statementType::attribute);
}
//======================================
