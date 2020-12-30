#ifndef OCCA_TESTS_PARSER_PARSERUTILS_HEADER
#define OCCA_TESTS_PARSER_PARSERUTILS_HEADER

#include <occa/internal/utils/testing.hpp>

#include <occa/internal/lang/expr.hpp>
#include <occa/internal/lang/parser.hpp>
#include <occa/internal/lang/builtins/types.hpp>
#include <occa/internal/lang/builtins/attributes.hpp>

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
  ASSERT_TRUE(parser.success);                  \
  {                                             \
    printer pout;                               \
    parser.root.print(pout);                    \
    std::cout << pout.str();                    \
  }


#define parseBadSource(str_)                    \
  parseSource(str_);                            \
  ASSERT_FALSE(parser.success);

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
  ASSERT_TRUE(parser.success);

#define setStatement(str_, type_)               \
  parseSource(str_);                            \
  ASSERT_EQ(1,                                  \
            parser.root.size());                \
  ASSERT_EQ_BINARY(type_,                       \
                   parser.root[0]->type());     \
  ASSERT_TRUE(parser.success);                  \
  statement = parser.root[0]
//======================================

class dummy : public attribute_t {
public:
  dummy() {}

  virtual const std::string& name() const {
    static std::string name_ = "dummy";
    return name_;
  }

  virtual bool forVariable() const {
    return true;
  }

  virtual bool forFunction() const {
    return true;
  }

  virtual bool forStatementType(const int sType) const {
    return true;
  }

  virtual bool isValid(const attributeToken_t &attr) const {
    return true;
  }
};

#endif
