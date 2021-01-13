#ifndef OCCA_INTERNAL_LANG_STATEMENT_STATEMENT_HEADER
#define OCCA_INTERNAL_LANG_STATEMENT_STATEMENT_HEADER

#include <map>
#include <vector>

#include <occa/internal/lang/attribute.hpp>
#include <occa/internal/lang/printer.hpp>
#include <occa/internal/lang/keyword.hpp>
#include <occa/internal/lang/token.hpp>
#include <occa/internal/lang/statement/statementArray.hpp>
#include <occa/internal/lang/expr/exprNodeArray.hpp>

namespace occa {
  namespace lang {
    class statement_t;
    class blockStatement;
    class ifStatement;
    class elifStatement;
    class elseStatement;
    class variableDeclaration;

    typedef std::map<statement_t*, exprNodeArray> statementExprMap;

    typedef std::vector<elifStatement*>      elifStatementVector;
    typedef std::vector<variableDeclaration> variableDeclarationVector;

    namespace statementType {
      extern const int none;
      extern const int all;

      extern const int empty;

      extern const int directive;
      extern const int pragma;
      extern const int comment;

      extern const int block;
      extern const int namespace_;

      extern const int function;
      extern const int functionDecl;

      extern const int class_;
      extern const int struct_;
      extern const int classAccess;

      extern const int enum_;
      extern const int union_;

      extern const int expression;
      extern const int declaration;

      extern const int goto_;
      extern const int gotoLabel;

      extern const int if_;
      extern const int elif_;
      extern const int else_;
      extern const int for_;
      extern const int while_;
      extern const int switch_;
      extern const int case_;
      extern const int default_;
      extern const int continue_;
      extern const int break_;

      extern const int return_;

      extern const int attribute;

      extern const int sourceCode;

      extern const int blockStatements;
    }

    class statement_t {
    public:
      blockStatement *up;
      token_t *source;
      attributeTokenMap attributes;

      statement_t(blockStatement *up_,
                  const token_t *source_);

      statement_t(blockStatement *up_,
                  const statement_t &other);

      virtual ~statement_t();

      template <class TM>
      inline bool is() const {
        return (dynamic_cast<const TM*>(this) != NULL);
      }

      template <class TM>
      inline TM& to() {
        TM *ptr = dynamic_cast<TM*>(this);
        OCCA_ERROR("Unable to cast statement_t::to",
                   ptr != NULL);
        return *ptr;
      }

      template <class TM>
      inline const TM& to() const {
        const TM *ptr = dynamic_cast<const TM*>(this);
        OCCA_ERROR("Unable to cast statement_t::to",
                   ptr != NULL);
        return *ptr;
      }

      statement_t& clone(blockStatement *up_ = NULL) const;
      virtual statement_t& clone_(blockStatement *up_) const = 0;

      static statement_t* clone(blockStatement *up_,
                                statement_t *smnt);

      virtual int type() const = 0;
      virtual std::string statementName() const = 0;

      void swapSource(statement_t &other);

      virtual bool hasInScope(const std::string &name);
      virtual keyword_t& getScopeKeyword(const std::string &name);

      void addAttribute(const attributeToken_t &attribute);
      bool hasAttribute(const std::string &attr) const;

      int childIndex() const;
      void removeFromParent();
      void replaceWith(statement_t &other);

      statementArray getParentPath();

      virtual statementArray getInnerStatements();

      exprNodeArray getExprNodes();

      virtual exprNodeArray getDirectExprNodes();

      void replaceExprNode(exprNode *currentNode, exprNode *newNode);
      virtual void safeReplaceExprNode(exprNode *currentNode, exprNode *newNode);

      void replaceKeyword(const keyword_t &currentKeyword, keyword_t &newKeyword);

      void replaceVariable(const variable_t &currentVar, variable_t &newVar);

      void replaceFunction(const function_t &currentFunc, function_t &newFunc);

      void replaceType(const type_t &currentType, type_t &newType);

      void updateVariableReferences();

      void updateIdentifierReferences();
      void updateIdentifierReferences(exprNode *expr);
      void updateIdentifierReferences(exprNodeArray &arr);

      void debugPrint() const;
      virtual void print(printer &pout) const = 0;

      std::string toString() const;
      operator std::string() const;

      void printWarning(const std::string &message) const;
      void printError(const std::string &message) const;
    };

    printer& operator << (printer &pout,
                          const statement_t &smnt);
  }
}

#endif
