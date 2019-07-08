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
          groupshared_q("groupshared", qualifierType::custom),
          constant_q("constant", qualifierType::custom) {
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
            var.add(0, groupshared_q);
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
        statementPtrVector kernelSmnts;
        findStatementsByAttr((statementType::functionDecl |
                              statementType::function),
                             "kernel",
                             root,
                             kernelSmnts);

        const int kernelCount = (int) kernelSmnts.size();
        for (int i = 0; i < kernelCount; ++i) {
          function_t *function;
          statement_t &kernelSmnt = *(kernelSmnts[i]);

          if (kernelSmnt.type() & statementType::functionDecl) {
            function = &(((functionDeclStatement&) kernelSmnt).function);
            migrateLocalDecls((functionDeclStatement&) kernelSmnt);
            if (!success) return;
          } else {
            function = &(((functionStatement&) kernelSmnt).function);
          }
          setKernelQualifiers(kernelSmnt, *function);
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

      void metalParser::setKernelQualifiers(statement_t &kernelSmnt,
                                            function_t &function) {
        function.returnType.add(0, kernel_q);

        const std::string &functionName = function.name();

        int argCount = (int) function.args.size();
        variablePtrVector constantArgs;
        for (int i = 0; i < argCount; ++i) {
          variable_t &arg = *(function.args[i]);
          arg.vartype = arg.vartype.flatten();
          if (arg.vartype.isPointerType()) {
            arg.add(0, device_q);
          } else {
            function.removeArgument(i--);
            --argCount;
            constantArgs.push_back(&arg);
          }
        }

        if (kernelSmnt.type() & statementType::functionDecl) {
          functionDeclStatement &kernelDeclSmnt = (
            (functionDeclStatement&) kernelSmnt
          );
          blockStatement &rootSmnt = *(kernelDeclSmnt.up);
          const int constantArgCount = (int) constantArgs.size();
          for (int i = constantArgCount - 1; i >= 0; --i) {
            variable_t &arg = *(constantArgs[i]);

            // Remove from scope before we update the name
            kernelDeclSmnt.removeFromScope(arg.name(), false);

            arg.setName(functionName + "_" + arg.name());
            arg.add(0, constant_q);
            arg.vartype.customSuffix = "[[function_constant(";
            arg.vartype.customSuffix += occa::toString(i);
            arg.vartype.customSuffix += ")]]";

            declarationStatement &declSmnt = *(
              new declarationStatement(&rootSmnt, NULL)
            );
            declSmnt.addDeclaration(arg);
            rootSmnt.addBefore(kernelDeclSmnt, declSmnt);
          }
        }

        variable_t occaGroupPositionArg(uint3, "_occa_group_position");
        variable_t occaThreadPositionArg(uint3, "_occa_thread_position");

        occaGroupPositionArg.vartype.customSuffix = (
          "[[threadgroup_position_in_grid]]"
        );
        occaThreadPositionArg.vartype.customSuffix = (
          "[[thread_position_in_threadgroup]]"
        );

        function.addArgument(occaGroupPositionArg);
        function.addArgument(occaThreadPositionArg);
      }
    }
  }
}
