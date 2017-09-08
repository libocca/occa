#if 0

#ifndef OCCA_PARSER_STATEMENT_HEADER2
#define OCCA_PARSER_STATEMENT_HEADER2

#include "occa/tools/gc.hpp"

namespace occa {
  class statementPrintInfo_t {
  public:
    std::string s;
    std::string indent;
    bool inlined;

    statementPrintInfo_t();

    void addIndentation();
    void removeIndentation();
  };

  class statement_t : public withRefs {
  public:
    parser_t &parser;
    scope_t scope;

    astNode_t root;
    statement_t *up;

    statement_t(parser_t &parser_);
    statement_t(const statement_t &s);
    statement_t& operator = (const statement_t &s);

    ~statement_t();

    virtual statement_t& clone() = 0;

    virtual void print(statementPrintInfo_t &pi) const = 0;

    void addIndentation(std::string &indent);
    void removeIndentation(std::string &indent);

    std::string toString() const;
    operator std::string() const;
    void print() const;
  };

  //---[ Directive ]--------------------
  class directiveStatement_t : public statement_t {
  public:
    std::string directive;

    virtual statement_t& clone() const;
    virtual void print(statementPrintInfo_t &pi) const;
  };
  //====================================

  //---[ Type ]-------------------------
  class typeStatement_t : public statement_t {
  public:

    virtual statement_t& clone() const;
    virtual void print(statementPrintInfo_t &pi) const;
  };
  //====================================

  //---[ Declaration ]------------------
  class declarationStatement_t : public statement_t {
  public:

    virtual statement_t& clone() const;
    virtual void print(statementPrintInfo_t &pi) const;
  };
  //====================================

  //---[ Expression ]-------------------
  class expressionStatement_t : public statement_t {
  public:

    virtual statement_t& clone() const;
    virtual void print(statementPrintInfo_t &pi) const;
  };
  //====================================

  //---[ Goto ]-------------------------
  class gotoStatement_t : public statement_t {
  public:
    std::string name;

    virtual statement_t& clone() const;
    virtual void print(statementPrintInfo_t &pi) const;
  };
  //====================================

  //---[ Block ]------------------------
  class blockStatement_t : public statement_t {
  public:
    std::vector<statement_t*> children;

    blockStatement_t();

    void addChild(statement_t &child);
    void clearChildren();

    virtual statement_t& clone() const;
    virtual void print(statementPrintInfo_t &pi) const;
    void printChildren(statementPrintInfo_t &pi) const;
  };
  //====================================

  //---[ Namespace ]--------------------
  class namespaceStatement_t : public blockStatement_t {
  public:
    std::string name;

    namespaceStatement_t(const std::string &name_);

    virtual statement_t& clone() const;
    virtual void print(statementPrintInfo_t &pi) const;
  };
  //====================================

  //---[ While ]------------------------
  class whileStatement_t : public blockStatement_t {
  public:
    statement &check;

    virtual statement_t& clone() const;
    virtual void print(statementPrintInfo_t &pi) const;
  };
  //====================================

  //---[ For ]--------------------------
  class forStatement_t : public blockStatement_t {
  public:
    statement &init, &check, &update;

    virtual statement_t& clone() const;
    virtual void print(statementPrintInfo_t &pi) const;
  };
  //====================================

  //---[ Switch ]-----------------------
  class switchStatement_t : public blockStatement_t {
  public:
    statement &value;

    virtual statement_t& clone() const;
    virtual void print(statementPrintInfo_t &pi) const;
  };
  //====================================

  //---[ Case ]-------------------------
  class caseStatement_t : public statement_t {
  public:
    statement &value;

    virtual statement_t& clone() const;
    virtual void print(statementPrintInfo_t &pi) const;
  };
  //====================================
}

#endif
#endif