#ifndef OCCA_LANG_STATEMENT_DPCPPSTATEMENT_HEADER
#define OCCA_LANG_STATEMENT_DPCPPSTATEMENT_HEADER

#include <occa/lang/statement/blockStatement.hpp>
#include <occa/lang/scope.hpp>

namespace occa
{
  namespace lang
  {
    class dpcppStatement : public blockStatement
    {
    public:
      statementPtrVector children;
      scope_t scope;

      dpcppStatement(blockStatement *up_,
                     token_t *source_);
      dpcppStatement(blockStatement *up_,
                     const dpcppStatement &other);
      virtual ~dpcppStatement();

      //      void copyFrom(const dpcppStatement &other);
      using statement_t::clone_;
      virtual statement_t &clone_(dpcppStatement *up_) const;

      virtual int type() const;
      virtual std::string statementName() const;

      virtual bool hasInScope(const std::string &name);
      virtual keyword_t &getScopeKeyword(const std::string &name);

      /*      bool addToScope(type_t &type,
                      const bool force = false);
      bool addToScope(function_t &func,
                      const bool force = false);
      bool addToScope(variable_t &var,
                      const bool force = false);

      void removeFromScope(const std::string &name,
                           const bool deleteSource = true);

      bool hasDirectlyInScope(const std::string &name);

      statement_t* operator [] (const int index);

      int size() const;

      void add(statement_t &child);

      bool add(statement_t &child,
               const int index);

      bool addFirst(statement_t &child);

      bool addLast(statement_t &child);

      bool addBefore(statement_t &child,
                     statement_t &newChild);

      bool addAfter(statement_t &child,
                    statement_t &newChild);

      void remove(statement_t &child);

      void set(statement_t &child);

      void swap(blockStatement &other);
      void swapScope(blockStatement &other);
      void swapChildren(blockStatement &other);

      void clear();

      exprNode* replaceIdentifiers(exprNode *expr);
*/
      virtual void print(printer &pout) const;
      void printChildren(printer &pout) const;
    };
  } // namespace lang
} // namespace occa

#endif
