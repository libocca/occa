#ifndef OCCA_INTERNAL_LANG_STATEMENT_BLOCKSTATEMENT_HEADER
#define OCCA_INTERNAL_LANG_STATEMENT_BLOCKSTATEMENT_HEADER

#include <occa/internal/lang/statement/statement.hpp>
#include <occa/internal/lang/scope.hpp>
#include <occa/internal/lang/statement/statementArray.hpp>

namespace occa {
  namespace lang {
    class blockStatement : public statement_t {
    public:
      statementArray children;
      scope_t scope;

      blockStatement(blockStatement *up_,
                     token_t *source_);
      blockStatement(blockStatement *up_,
                     const blockStatement &other);
      virtual ~blockStatement();

      void copyFrom(const blockStatement &other);

      virtual statement_t& clone_(blockStatement *up_) const;

      virtual int type() const;
      virtual std::string statementName() const;

      virtual bool hasInScope(const std::string &name);
      virtual keyword_t& getScopeKeyword(const std::string &name);
      type_t* getScopeType(const std::string &name);

      bool addToScope(type_t &type,
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

      virtual void print(printer &pout) const;
      void printChildren(printer &pout) const;
    };
  }
}

#endif
