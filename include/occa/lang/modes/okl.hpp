#ifndef OCCA_LANG_MODES_OKL_HEADER
#define OCCA_LANG_MODES_OKL_HEADER

#include <vector>

#include <occa/lang/statement.hpp>

namespace occa {
  namespace lang {
    class leftUnaryOpNode;
    class rightUnaryOpNode;
    class binaryOpNode;

    namespace transforms {
      class smntTreeNode;
    }

    namespace okl {
      bool checkKernels(statement_t &root);

      bool checkKernel(functionDeclStatement &kernelSmnt);

      //---[ Declaration ]--------------
      bool checkLoops(functionDeclStatement &kernelSmnt);

      bool checkForDoubleLoops(statementPtrVector &loopSmnts,
                               const std::string &badAttr);

      bool checkOklForStatements(functionDeclStatement &kernelSmnt,
                                 statementPtrVector &forSmnts,
                                 const std::string &attrName);
      //================================

      //---[ Loop Logic ]---------------
      bool oklLoopMatcher(statement_t &smnt);
      bool oklDeclAttrMatcher(statement_t &smnt,
                              const std::string &attr);
      bool oklAttrMatcher(statement_t &smnt,
                          const std::string &attr);
      bool oklSharedMatcher(statement_t &smnt);
      bool oklExclusiveMatcher(statement_t &smnt);

      bool checkLoopOrders(functionDeclStatement &kernelSmnt);

      bool checkLoopOrder(transforms::smntTreeNode &root);
      bool checkLoopType(transforms::smntTreeNode &node,
                         int &outerCount,
                         int &innerCount);
      //================================

      //---[ Type Logic ]---------------
      bool checkSharedOrder(transforms::smntTreeNode &root);
      bool checkExclusiveOrder(transforms::smntTreeNode &root);
      bool checkOKLTypeInstance(statement_t &typeSmnt,
                                const std::string &attr);
      bool checkValidSharedArray(statement_t &smnt);
      //================================

      //---[ Skip Logic ]---------------
      bool checkBreakAndContinue(functionDeclStatement &kernelSmnt);
      //================================

      //---[ Transformations ]----------
      void addAttributes(parser_t &parser);

      void setLoopIndices(functionDeclStatement &kernelSmnt);

      void setForLoopIndex(forStatement &forSmnt,
                           const std::string &attr);
      //================================
    }
  }
}

#endif
