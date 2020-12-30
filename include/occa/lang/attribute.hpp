#ifndef OCCA_INTERNAL_LANG_ATTRIBUTE_HEADER
#define OCCA_INTERNAL_LANG_ATTRIBUTE_HEADER

#include <iostream>
#include <vector>
#include <map>

#include <occa/internal/io/output.hpp>

namespace occa {
  namespace lang {
    class parser_t;
    class identifierToken;
    class attribute_t;
    class attributeToken_t;
    class attributeArg_t;
    class exprNode;
    class vartype_t;
    class variable_t;
    class function_t;
    class statement_t;
    class expressionStatement;

    typedef std::vector<attributeArg_t>             attributeArgVector;
    typedef std::map<std::string, attributeArg_t>   attributeArgMap;
    typedef std::map<std::string, attributeToken_t> attributeTokenMap;
    typedef std::map<std::string, attribute_t*>     nameToAttributeMap;

    //---[ Attribute Type ]-------------
    class attribute_t {
    public:
      virtual ~attribute_t();

      virtual const std::string& name() const = 0;

      virtual bool forVariable() const;
      virtual bool forFunction() const;
      virtual bool forStatementType(const int sType) const = 0;

      virtual bool isValid(const attributeToken_t &attr) const = 0;
    };
    //==================================

    //---[ Attribute Arg ]--------------
    class attributeArg_t {
    public:
      exprNode *expr;
      attributeTokenMap attributes;

      attributeArg_t();

      attributeArg_t(exprNode *expr_);

      attributeArg_t(exprNode *expr_,
                     attributeTokenMap attributes_);

      attributeArg_t(const attributeArg_t &other);

      attributeArg_t& operator = (const attributeArg_t &other);

      ~attributeArg_t();

      void clear();

      bool exists() const;
    };
    //==================================

    //---[ Attribute ]------------------
    class attributeToken_t {
    public:
      const attribute_t *attrType;
      identifierToken *source;
      attributeArgVector args;
      attributeArgMap kwargs;

      attributeToken_t();
      attributeToken_t(const attribute_t &attrType_,
                       identifierToken &source_);
      attributeToken_t(const attributeToken_t &other);
      attributeToken_t& operator = (const attributeToken_t &other);
      virtual ~attributeToken_t();

      void copyFrom(const attributeToken_t &other);
      void clear();

      const std::string& name() const;

      bool forVariable() const;
      bool forFunction() const;
      bool forStatementType(const int sType) const;

      attributeArg_t* operator [] (const int index);
      attributeArg_t* operator [] (const std::string &arg);

      void printWarning(const std::string &message) const;
      void printError(const std::string &message) const;
    };
    //==================================

    io::output& operator << (io::output &out,
                             const attributeArg_t &attr);

    io::output& operator << (io::output &out,
                             const attributeToken_t &attr);

    io::output& operator << (io::output &out,
                             const attributeTokenMap &attributes);
  }
}

#endif
