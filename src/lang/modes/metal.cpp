#include <occa/tools/string.hpp>
#include <occa/lang/modes/metal.hpp>
#include <occa/lang/modes/okl.hpp>
#include <occa/lang/modes/oklForStatement.hpp>
#include <occa/lang/builtins/attributes.hpp>
#include <occa/lang/builtins/types.hpp>

namespace occa {
  namespace lang {
    namespace okl {
      metalParser::metalParser(const occa::properties &settings_) :
          withLauncher(settings_),
          kernel_q("kernel", qualifierType::custom),
          device_q("device", qualifierType::custom),
          shared_q("groupshared", qualifierType::custom) {
        okl::addAttributes(*this);
      }

      void metalParser::onClear() {
        launcherClear();
      }

      void metalParser::beforePreprocessing() {
        preprocessor.addCompilerDefine("OCCA_USING_GPU", "1");
      }

      void metalParser::beforeKernelSplit() {
        if (!success) return;
        setSharedQualifiers();
      }

      void metalParser::afterKernelSplit() {
        addBarriers();

        if (!success) return;
        setupKernels();
      }

      std::string metalParser::getOuterIterator(const int loopIndex) {
        // [0, 1, 2] -> ['x', 'y', 'z']
        std::string name = "_occa_group_position.";
        name += ('x' + (char) loopIndex);
        return name;
      }

      std::string metalParser::getInnerIterator(const int loopIndex) {
        // [0, 1, 2] -> ['x', 'y', 'z']
        std::string name = "_occa_thread_position.";
        name += ('x' + (char) loopIndex);
        return name;
      }

      void metalParser::setSharedQualifiers() {
        statementExprMap exprMap;
        findStatements(statementType::declaration,
                       exprNodeType::variable,
                       root,
                       sharedVariableMatcher,
                       exprMap);

        statementExprMap::iterator it = exprMap.begin();
        while (it != exprMap.end()) {
          declarationStatement &declSmnt = *((declarationStatement*) it->first);
          const int declCount = declSmnt.declarations.size();
          for (int i = 0; i < declCount; ++i) {
            variable_t &var = *(declSmnt.declarations[i].variable);
            if (!var.hasAttribute("shared")) {
              continue;
            }
            var.add(0, shared_q);
          }
          ++it;
        }
      }

      bool metalParser::sharedVariableMatcher(exprNode &expr) {
        return expr.hasAttribute("shared");
      }

      void metalParser::addBarriers() {
        statementPtrVector statements;
        findStatementsByAttr(statementType::empty,
                             "barrier",
                             root,
                             statements);

        const int count = (int) statements.size();
        for (int i = 0; i < count; ++i) {
          // TODO 1.1: Implement proper barriers
          emptyStatement &smnt = *((emptyStatement*) statements[i]);

          statement_t &barrierSmnt = (
            *(new expressionStatement(
                smnt.up,
                *(new identifierNode(smnt.source,
                                     "threadgroup_barrier(mem_threadgroup)"))
              ))
          );

          smnt.up->addBefore(smnt,
                             barrierSmnt);

          smnt.up->remove(smnt);
          delete &smnt;
        }
      }

      void metalParser::setupKernels() {
        // Need to add the following args at the end:
        //   _occa_group_position [[threadgroup_position_in_grid]]
        //   _occa_thread_position [[thread_position_in_threadgroup]]
        statementPtrVector kernelSmnts;
        findStatementsByAttr((statementType::functionDecl |
                              statementType::function),
                             "kernel",
                             root,
                             kernelSmnts);

        const int kernelCount = (int) kernelSmnts.size();
        for (int i = 0; i < kernelCount; ++i) {
          function_t *function;
          if (kernelSmnts[i]->type() & statementType::functionDecl) {
            function = &(((functionDeclStatement*) kernelSmnts[i])->function);

            migrateLocalDecls(*((functionDeclStatement*) kernelSmnts[i]));
            if (!success) return;
          } else {
            function = &(((functionStatement*) kernelSmnts[i])->function);
          }
          setKernelQualifiers(*function);
          if (!success) return;
        }
      }

      void metalParser::migrateLocalDecls(functionDeclStatement &kernelSmnt) {
        statementExprMap exprMap;
        findStatements(statementType::declaration,
                       exprNodeType::variable,
                       kernelSmnt,
                       sharedVariableMatcher,
                       exprMap);

        statementExprMap::iterator it = exprMap.begin();
        while (it != exprMap.end()) {
          declarationStatement &declSmnt = *((declarationStatement*) it->first);
          variable_t *var = declSmnt.declarations[0].variable;

          if (var->hasAttribute("shared")) {
            declSmnt.removeFromParent();
            kernelSmnt.addFirst(declSmnt);
          }
          ++it;
        }
      }

      void metalParser::setKernelQualifiers(function_t &function) {
        function.returnType.add(0, kernel_q);

        const int argCount = (int) function.args.size();
        for (int ai = 0; ai < argCount; ++ai) {
          variable_t &arg = *(function.args[ai]);
          arg.vartype = arg.vartype.flatten();
          if (arg.vartype.isPointerType()) {
            arg.add(0, device_q);
          }
        }
      }
    }
  }
}
