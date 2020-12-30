#ifndef OCCA_INTERNAL_LANG_MODES_OKL_HEADER
#define OCCA_INTERNAL_LANG_MODES_OKL_HEADER

#include <functional>
#include <vector>

#include <occa/internal/lang/statement.hpp>

namespace occa {
  namespace lang {
    class forStatement;

    namespace okl {
      typedef std::vector<statementArray> statementArrayVector;

      typedef std::function<void (forStatement &forSmnt, const std::string &attr, const statementArray &path)> oklForVoidCallback;

      bool kernelsAreValid(blockStatement &root);

      bool kernelIsValid(functionDeclStatement &kernelSmnt);

      bool kernelHasValidReturnType(functionDeclStatement &kernelSmnt);

      bool kernelHasValidOklLoops(functionDeclStatement &kernelSmnt);

      bool outerLoopHasValidOklLoopOrdering(forStatement &outerMostForSmnt,
                                            statementArrayVector &loopPaths);

      bool pathHasValidOklLoopOrdering(statementArray &loopPath,
                                       int &outerLoopCount,
                                       int &innerLoopCount);

      bool kernelHasValidSharedAndExclusiveDeclarations(functionDeclStatement &kernelSmnt);

      bool hasProperSharedArrayDeclaration(variable_t &var);

      bool hasProperSharedOrExclusiveUsage(statement_t *smnt,
                                           const std::string &attrName,
                                           bool varIsBeingDeclared);

      bool kernelHasValidLoopBreakAndContinue(functionDeclStatement &kernelSmnt);

      //---[ Helper Methods ]-----------
      bool isOklForLoop(statement_t *smnt);

      bool isOklForLoop(statement_t *smnt, std::string &oklAttr);

      void forOklForLoopStatements(statement_t &root, oklForVoidCallback func);
      //================================

      //---[ Transformations ]----------
      void addOklAttributes(parser_t &parser);

      void setOklLoopIndices(functionDeclStatement &kernelSmnt);
      //================================
    }
  }
}

#endif
