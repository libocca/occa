#include <occa/tools/string.hpp>
#include <occa/lang/modes/cuda.hpp>
#include <occa/lang/modes/okl.hpp>
#include <occa/lang/modes/oklForStatement.hpp>
#include <occa/lang/builtins/attributes.hpp>
#include <occa/lang/builtins/types.hpp>
#include <occa/lang/expr/identifierNode.hpp>

namespace occa {
  namespace lang {
    namespace okl {
      cudaParser::cudaParser(const occa::properties &settings_) :
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
        root.children
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
        root.children
            .flatFilterByStatementType(statementType::empty, "barrier")
            .forEach([&](statement_t *smnt) {
                // TODO: Implement proper barriers
                emptyStatement &emptySmnt = *((emptyStatement*) smnt);

                statement_t &barrierSmnt = (
                  *(new expressionStatement(
                      emptySmnt.up,
                      *(new identifierNode(emptySmnt.source,
                                           " __syncthreads()"))
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
    }
  }
}
