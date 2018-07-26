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
#ifndef OCCA_TESTS_PARSER_PARSERUTILS_HEADER
#define OCCA_TESTS_PARSER_PARSERUTILS_HEADER

#include <occa/tools/testing.hpp>

#include <occa/lang/exprNode.hpp>
#include <occa/lang/parser.hpp>
#include <occa/lang/builtins/types.hpp>
#include <occa/lang/builtins/attributes.hpp>

using namespace occa::lang;

//---[ Util Methods ]-------------------
std::string source;

#ifndef OCCA_TEST_PARSER_TYPE
#  define OCCA_TEST_PARSER_TYPE parser_t
#endif
OCCA_TEST_PARSER_TYPE parser;

void setSource(const std::string &s) {
  source = s;
  parser.setSource(source, false);
}

void parseSource(const std::string &s) {
  source = s;
  parser.parseSource(source);
}

#define parseAndPrintSource(str_)               \
  parseSource(str_);                            \
  ASSERT_TRUE(parser.success)                   \
  {                                             \
    printer pout;                               \
    parser.root.print(pout);                    \
    std::cout << pout.str();                    \
  }


#define parseBadSource(str_)                    \
  parseSource(str_);                            \
  ASSERT_FALSE(parser.success)

template <class smntType>
smntType& getStatement(const int index = 0) {
  return parser.root[index]->to<smntType>();
}
//======================================

//---[ Macro Util Methods ]-------------
#define testStatementPeek(str_, type_)          \
  setSource(str_);                              \
  ASSERT_EQ_BINARY(type_,                       \
                   parser.peek());              \
  ASSERT_TRUE(parser.success)

#define setStatement(str_, type_)               \
  parseSource(str_);                            \
  ASSERT_EQ(1,                                  \
            parser.root.size());                \
  ASSERT_EQ_BINARY(type_,                       \
                   parser.root[0]->type())      \
  ASSERT_TRUE(parser.success);                  \
  statement = parser.root[0]
//======================================

class dummy : public attribute_t {
public:
  dummy() {}

  virtual std::string name() const {
    return "dummy";
  }

  virtual bool forVariable() const {
    return true;
  }

  virtual bool forFunction() const {
    return true;
  }

  virtual bool forStatement(const int sType) const {
    return true;
  }

  virtual bool isValid(const attributeToken_t &attr) const {
    return true;
  }
};

#endif
