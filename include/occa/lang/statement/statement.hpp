#ifndef OCCA_LANG_STATEMENT_STATEMENT_HEADER
#define OCCA_LANG_STATEMENT_STATEMENT_HEADER

#include <vector>

#include <occa/lang/attribute.hpp>
#include <occa/lang/printer.hpp>
#include <occa/lang/keyword.hpp>
#include <occa/lang/token.hpp>

namespace occa {
  namespace lang {
    class statement_t;
    class blockStatement;
    class ifStatement;
    class elifStatement;
    class elseStatement;
    class variableDeclaration;

    typedef std::vector<statement_t*>        statementPtrVector;
    typedef std::vector<elifStatement*>      elifStatementVector;
    typedef std::vector<variableDeclaration> variableDeclarationVector;

    namespace statementType {
      extern const int none;
      extern const int all;

      extern const int empty;

      extern const int directive;
      extern const int pragma;

      extern const int block;
      extern const int namespace_;

      extern const int typeDecl;
      extern const int function;
      extern const int functionDecl;
      extern const int classAccess;

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

      virtual void print(printer &pout) const = 0;

      std::string toString() const;
      operator std::string() const;
      void debugPrint() const;

      void printWarning(const std::string &message) const;
      void printError(const std::string &message) const;
    };

    printer& operator << (printer &pout,
                          const statement_t &smnt);
  }
}

#endif
