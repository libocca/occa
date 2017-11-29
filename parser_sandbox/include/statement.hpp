#ifndef OCCA_PARSER_STATEMENT_HEADER2
#define OCCA_PARSER_STATEMENT_HEADER2

#include "occa/tools/gc.hpp"
#include "macro.hpp"
#include "context.hpp"
#include "scope.hpp"

namespace occa {
  namespace lang {
    class directiveStatement_t;
    class blockStatement_t;
    class typeDeclStatement_t;
    class classAccessStatement_t;
    class expressionStatement_t;
    class declarationStatement_t;
    class gotoStatement_t;
    class gotoLabelStatement_t;
    class namespaceStatement_t;
    class whileStatement_t;
    class forStatement_t;
    class switchStatement_t;
    class caseStatement_t;

    class statementType {
    public:
      static const int none        = 0;
      static const int block       = (1 << 0);
      static const int typeDecl    = (1 << 1);
      static const int expression  = (1 << 2);
      static const int declaration = (1 << 3);
      static const int while_      = (1 << 4);
      static const int for_        = (1 << 5);
      static const int switch_     = (1 << 6);
    };

    class statement_t : public withRefs {
    public:
      statement_t *up;
      context_t &context;
      scope_t scope;

      statement_t(context_t &context_);

      virtual ~statement_t();

      virtual statement_t& clone() const = 0;

      virtual int type() const;
      virtual bool hasScope() const;

      // Creation methods
      directiveStatement_t   newDirectiveStatement(macro_t &macro);
      blockStatement_t       newBlockStatement();
      typeDeclStatement_t    newTypeDeclarationStatement(declarationType_t &declType_);
      classAccessStatement_t newClassAccessStatement(const int access_);
      expressionStatement_t  newExpressionStatement();
      declarationStatement_t newDeclarationStatement();
      gotoStatement_t        newGotoStatement(const std::string &name_);
      gotoLabelStatement_t   newGotoLabelStatement(const std::string &name_);
      namespaceStatement_t   newNamespaceStatement(const std::string &name_);
      whileStatement_t       newWhileStatement(statement_t &check_);
      forStatement_t         newForStatement(statement_t &init_,
                                             statement_t &check_,
                                             statement_t &update_);
      switchStatement_t      newSwitchStatement(statement_t &value_);
      caseStatement_t        newCaseStatement(statement_t &value_);

      virtual void print(printer_t &pout) const = 0;

      std::string toString() const;
      operator std::string() const;
      void print() const;
    };

    //---[ Directive ]--------------------
    class directiveStatement_t : public statement_t {
    public:
      macro_t &macro;

      directiveStatement_t(context_t &context_,
                           macro_t &macro_);

      virtual statement_t& clone() const;
      virtual void print(printer_t &pout) const;
    };
    //====================================

    //---[ Block ]------------------------
    class blockStatement_t : public statement_t {
    public:
      std::vector<statement_t*> children;

      blockStatement_t(context_t &context_);

      void addChild(statement_t &child);
      void clearChildren();

      virtual statement_t& clone() const;

      virtual int type() const;

      virtual bool hasScope() const;

      virtual void print(printer_t &pout) const;
      void printChildren(printer_t &pout) const;
    };
    //====================================

    //---[ Type ]-------------------------
    class typeDeclStatement_t : public statement_t {
    public:
      declarationType_t &declType;

      typeDeclStatement_t(context_t &context_,
                          declarationType_t &declType_);

      virtual statement_t& clone() const;

      virtual int type() const;

      virtual bool hasScope() const;

      virtual void print(printer_t &pout) const;
    };

    class classAccessStatement_t : public statement_t {
    public:
      int access;

      classAccessStatement_t(context_t &context_,
                             const int access_);

      virtual statement_t& clone() const;
      virtual void print(printer_t &pout) const;
    };
    //====================================

    //---[ Expression ]------------------- TODO
    class expressionStatement_t : public statement_t {
    public:
      expressionStatement_t(context_t &context_);

      virtual statement_t& clone() const;

      virtual int type() const;

      virtual void print(printer_t &pout) const;
    };

    class declarationStatement_t : public statement_t {
    public:
      declarationStatement_t(context_t &context_);

      virtual statement_t& clone() const;

      virtual int type() const;

      virtual void print(printer_t &pout) const;
    };
    //====================================

    //---[ Goto ]-------------------------
    class gotoStatement_t : public statement_t {
    public:
      std::string name;

      gotoStatement_t(context_t &context_,
                      const std::string &name_);

      virtual statement_t& clone() const;
      virtual void print(printer_t &pout) const;
    };

    class gotoLabelStatement_t : public statement_t {
    public:
      std::string name;

      gotoLabelStatement_t(context_t &context_,
                           const std::string &name_);

      virtual statement_t& clone() const;
      virtual void print(printer_t &pout) const;
    };
    //====================================

    //---[ Namespace ]--------------------
    class namespaceStatement_t : public blockStatement_t {
    public:
      std::string name;

      namespaceStatement_t(context_t &context_,
                           const std::string &name_);

      virtual bool hasScope() const;

      virtual statement_t& clone() const;
      virtual void print(printer_t &pout) const;
    };
    //====================================

    //---[ While ]------------------------
    class whileStatement_t : public blockStatement_t {
    public:
      statement_t &check;

      whileStatement_t(context_t &context_,
                       statement_t &check_);

      virtual statement_t& clone() const;

      virtual int type() const;

      virtual bool hasScope() const;

      virtual void print(printer_t &pout) const;
    };
    //====================================

    //---[ For ]--------------------------
    class forStatement_t : public blockStatement_t {
    public:
      statement_t &init, &check, &update;

      forStatement_t(context_t &context_,
                     statement_t &init_,
                     statement_t &check_,
                     statement_t &update_);

      virtual statement_t& clone() const;

      virtual int type() const;

      virtual bool hasScope() const;

      virtual void print(printer_t &pout) const;
    };
    //====================================

    //---[ Switch ]-----------------------
    class switchStatement_t : public blockStatement_t {
    public:
      statement_t &value;

      switchStatement_t(context_t &context_,
                        statement_t &value_);

      virtual statement_t& clone() const;

      virtual int type() const;

      virtual bool hasScope() const;

      virtual void print(printer_t &pout) const;
    };
    //====================================

    //---[ Case ]-------------------------
    class caseStatement_t : public statement_t {
    public:
      statement_t &value;

      caseStatement_t(context_t &context_,
                      statement_t &value_);

      virtual statement_t& clone() const;
      virtual void print(printer_t &pout) const;
    };
    //====================================
  }
}

#endif
