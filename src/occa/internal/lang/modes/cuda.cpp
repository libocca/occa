#include <occa/internal/utils/string.hpp>
#include <occa/internal/lang/modes/cuda.hpp>
#include <occa/internal/lang/modes/okl.hpp>
#include <occa/internal/lang/modes/oklForStatement.hpp>
#include <occa/internal/lang/expr.hpp>
#include <occa/internal/lang/builtins/attributes.hpp>
#include <occa/internal/lang/builtins/types.hpp>
#include <occa/internal/lang/builtins/attributes/atomic.hpp>

namespace occa {
  namespace lang {
    namespace okl {
      cudaParser::cudaParser(const occa::json &settings_) :
        withLauncher(settings_),
        constant("__constant__", qualifierType::custom),
        global("__global__", qualifierType::custom),
        device("__device__", qualifierType::custom),
        shared("__shared__", qualifierType::custom) {

        okl::addOklAttributes(*this);
      }

      void cudaParser::onClear() {
        launcherClear();
      }

      void cudaParser::beforePreprocessing() {
        preprocessor.addCompilerDefine("OCCA_USING_GPU", "1");
      }

      void cudaParser::beforeKernelSplit() {
        updateConstToConstant();

        if (!success) return;
        setFunctionQualifiers();

        if (!success) return;
        setSharedQualifiers();
      }

      void cudaParser::afterKernelSplit() {
        addBarriers();

        if (!success) return;
        setupKernels();

        if (!success) return;
        setupAtomics();
      }

      std::string cudaParser::getOuterIterator(const int loopIndex) {
        std::string name = "blockIdx.";
        name += 'x' + (char) loopIndex;
        return name;
      }

      std::string cudaParser::getInnerIterator(const int loopIndex) {
        std::string name = "threadIdx.";
        name += 'x' + (char) loopIndex;
        return name;
      }

      std::string cudaParser::launchBoundsAttribute(const int innerDims[3]) {
        const int innerTotal{innerDims[0]*innerDims[1]*innerDims[2]};
        const std::string lbAttr = "__launch_bounds__(" + std::to_string(innerTotal) + ")";
        return lbAttr;
      }

      void cudaParser::updateConstToConstant() {
        root.children
          .forEachDeclaration([&](variableDeclaration &decl) {
            variable_t &var = decl.variable();

            if (var.has(const_) && !var.has(typedef_)) {
              var -= const_;
              var += constant;
            }
          });
      }

      void cudaParser::setFunctionQualifiers() {
        root.children
          .filterByStatementType(statementType::functionDecl)
          .forEach([&](statement_t *smnt) {
            functionDeclStatement &funcDeclSmnt = (functionDeclStatement&) *smnt;

            // Only add __device__ to non-kernel functions
            if (funcDeclSmnt.hasAttribute("kernel")) {
              return;
            }

            vartype_t &vartype = funcDeclSmnt.function().returnType;
            vartype.qualifiers.addFirst(vartype.origin(),
                                        device);
          });
      }

      void cudaParser::setSharedQualifiers() {
        statementArray::from(root)
          .flatFilterByStatementType(statementType::declaration)
          .forEach([&](statement_t *smnt) {
            declarationStatement &declSmnt = *((declarationStatement*) smnt);
            for (variableDeclaration &varDecl : declSmnt.declarations) {
              variable_t &var = varDecl.variable();
              if (var.hasAttribute("shared")) {
                var += shared;
              }
            }
          });
      }

      void cudaParser::addBarriers() {
        statementArray::from(root)
          .flatFilterByStatementType(statementType::empty, "barrier")
          .forEach([&](statement_t *smnt) {
            // TODO: Implement proper barriers
            emptyStatement &emptySmnt = *((emptyStatement*) smnt);

            const attributes::barrier::SyncType syncType = (
              emptySmnt.hasAttribute("barrier")
              ? attributes::barrier::getBarrierSyncType(
                &emptySmnt.attributes["barrier"]
              )
              : attributes::barrier::syncDefault
            );

            std::string source;
            switch (syncType) {
              case attributes::barrier::invalid:
              case attributes::barrier::syncDefault:
                source = "__syncthreads();";
                break;

              case attributes::barrier::syncWarp:
                source = "__syncwarp();";
                break;
            }

            statement_t &barrierSmnt = (
              *(new sourceCodeStatement(
                  emptySmnt.up,
                  emptySmnt.source,
                  source
                ))
            );

            emptySmnt.replaceWith(barrierSmnt);
            delete &emptySmnt;
          });
      }

      void cudaParser::setupKernels() {
        root.children
          .forEachKernelStatement([&](functionDeclStatement &kernelSmnt) {
            // Set kernel qualifiers
           vartype_t &vartype = kernelSmnt.function().returnType;

           vartype.qualifiers.addFirst(vartype.origin(),
                                        global);
           vartype.qualifiers.addFirst(vartype.origin(),
                                        externC);
          });
      }

      void cudaParser::setupAtomics() {
        success &= attributes::atomic::applyCodeTransformation(
          root,
          transformAtomicBlockStatement,
          transformAtomicBasicExpressionStatement
        );
      }

      bool cudaParser::transformAtomicBlockStatement(blockStatement &blockSmnt) {
        blockSmnt.printError("Unable to transform general @atomic code");
        return false;
      }

      bool cudaParser::transformAtomicBasicExpressionStatement(expressionStatement &exprSmnt) {
        // TODO: Create actual statements + expressions, don't just shove strings in
        const opType_t &opType = expr(exprSmnt.expr).opType();

        printer pout;
        if (opType & operatorType::unary) {
          // Cases:
          //   @atomic --i;
          //   @atomic i--;
          //   @atomic ++i;
          //   @atomic i++;
          expr variable;
          if (opType & operatorType::leftUnary) {
            variable = ((leftUnaryOpNode*) exprSmnt.expr)->value;
          }
          else if (opType & operatorType::rightUnary) {
            variable = ((rightUnaryOpNode*) exprSmnt.expr)->value;
          }
          // Build source code
          if (opType & operatorType::increment) {
            pout << "atomicInc(&" << expr::parens(variable) << ");";
          } else if (opType & operatorType::decrement) {
            pout << "atomicDec(&" << expr::parens(variable) << ");";
          } else {
            exprSmnt.printError("Unable to transform @atomic code");
            return false;
          }
        }
        else if (opType & operatorType::binary) {
          // Cases:
          //   @atomic i += 1;
          //   @atomic i -= 1;
          //   @atomic i &= 1;
          //   @atomic i |= 1;
          //   @atomic i ^= 1;
          binaryOpNode &binaryNode = (binaryOpNode&) *exprSmnt.expr;
          expr variable = binaryNode.leftValue;
          expr value = binaryNode.rightValue;
          if (opType & operatorType::addEq) {
            pout << "atomicAdd(&" << expr::parens(variable) << ", " << value << ");";
          } else if (opType & operatorType::subEq) {
            pout << "atomicSub(&" << expr::parens(variable) << ", " << value << ");";
          } else if (opType & operatorType::andEq) {
            pout << "atomicAnd(&" << expr::parens(variable) << ", " << value << ");";
          } else if (opType & operatorType::orEq) {
            pout << "atomicOr(&" << expr::parens(variable) << ", " << value << ");";
          } else if (opType & operatorType::xorEq) {
            pout << "atomicXor(&" << expr::parens(variable) << ", " << value << ");";
          } else {
            exprSmnt.printError("Unable to transform @atomic code");
            return false;
          }
        }

        statement_t &atomicSmnt = (
          *(new sourceCodeStatement(
              exprSmnt.up,
              exprSmnt.source,
              pout.str()
            ))
        );

        exprSmnt.replaceWith(atomicSmnt);
        delete &exprSmnt;

        return true;
      }
    }
  }
}
