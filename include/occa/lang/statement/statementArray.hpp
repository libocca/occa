#ifndef OCCA_INTERNAL_LANG_STATEMENT_STATEMENTARRAY_HEADER
#define OCCA_INTERNAL_LANG_STATEMENT_STATEMENTARRAY_HEADER

#include <functional>

#include <occa/internal/lang/utils/array.hpp>
#include <occa/internal/lang/expr/exprNodeArray.hpp>

namespace occa {
  namespace lang {
    class statement_t;
    class blockStatement;
    class declarationStatement;
    class functionDeclStatement;
    class variableDeclaration;

    class statementArray : public array<statement_t*> {
     public:
      typedef std::function<statementArray (statement_t *smnt)> smntMapCallback;
      typedef std::function<statementArray (statement_t *smnt, const statementArray &path)> smntWithPathMapCallback;

      typedef std::function<bool (statement_t *smnt)> smntFilterCallback;
      typedef std::function<bool (statement_t *smnt, const statementArray &path)> smntWithPathFilterCallback;

      typedef std::function<void (statement_t *smnt)> smntVoidCallback;
      typedef std::function<void (statement_t *smnt, const statementArray &path)> smntWithPathVoidCallback;

      typedef std::function<void (functionDeclStatement &kernelSmnt)> functionDeclVoidCallback;
      typedef std::function<void (variableDeclaration &decl)> declarationVoidCallback;
      typedef std::function<void (variableDeclaration &decl, declarationStatement &declSmnt)> declarationWithSmntVoidCallback;

      OCCA_LANG_ARRAY_DEFINE_METHODS(statementArray, statement_t*)

      static statementArray from(statement_t &smnt);

      statementArray flatMap(smntMapCallback func) const;
      exprNodeArray flatMap(smntExprMapCallback func) const;
      statementArray flatMap(smntWithPathMapCallback func) const;

      statementArray flatFilter(smntFilterCallback func) const;
      exprNodeArray flatFilter(smntExprFilterCallback func) const;
      statementArray flatFilter(smntWithPathFilterCallback func) const;

      void nestedForEach(smntVoidCallback func) const;
      void nestedForEach(smntExprVoidCallback func) const;
      void nestedForEach(smntWithPathVoidCallback func) const;

      // Filter helper functions
      statementArray filterByStatementType(const int smntType) const;

      statementArray filterByStatementType(const int smntType, const std::string &attr) const;

      statementArray flatFilterByStatementType(const int smntType) const;

      statementArray flatFilterByStatementType(const int smntType, const std::string &attr) const;

      statementArray filterByAttribute(const std::string &attr) const;

      statementArray flatFilterByAttribute(const std::string &attr) const;

      exprNodeArray flatFilterByExprType(const int allowedExprNodeType) const;

      exprNodeArray flatFilterByExprType(const int allowedExprNodeType, const std::string &attr) const;

      exprNodeArray flatFilterExprNodesByTypes(const int smntType,
                                               const int allowedExprNodeType) const;

      //---[ Helper Methods ]-------------------------------
      //  |---[ Iteration Methods ]-------------------------
      void iterateStatements(smntVoidCallback func) const;

      void iterateStatements(smntWithPathVoidCallback func) const;

      void iterateStatement(smntWithPathVoidCallback func, statementArray &path, statement_t &smnt) const;

      void iterateExprNodes(smntExprVoidCallback func) const;
      //  |=================================================

      //  |---[ Transformation methods ]--------------------
      statementArray getKernelStatements() const;

      void forEachKernelStatement(functionDeclVoidCallback func) const;

      void forEachDeclaration(declarationVoidCallback func) const;
      void forEachDeclaration(declarationWithSmntVoidCallback func) const;

      void nestedForEachDeclaration(declarationVoidCallback func) const;
      void nestedForEachDeclaration(declarationWithSmntVoidCallback func) const;
      //  |=================================================
      //====================================================
    };
  }
}

#endif
