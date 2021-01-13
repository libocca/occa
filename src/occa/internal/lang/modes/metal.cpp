#include <occa/internal/utils/string.hpp>
#include <occa/internal/lang/modes/metal.hpp>
#include <occa/internal/lang/modes/okl.hpp>
#include <occa/internal/lang/modes/oklForStatement.hpp>
#include <occa/internal/lang/builtins/attributes.hpp>
#include <occa/internal/lang/builtins/types.hpp>
#include <occa/internal/lang/expr.hpp>

namespace occa {
  namespace lang {
    namespace okl {
      metalParser::metalParser(const occa::json &settings_) :
          withLauncher(settings_),
          kernel_q("kernel", qualifierType::custom),
          device_q("device", qualifierType::custom),
          threadgroup_q("threadgroup", qualifierType::custom),
          constant_q("constant", qualifierType::custom) {

        okl::addOklAttributes(*this);
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
        statementArray::from(root)
            .nestedForEachDeclaration([&](variableDeclaration &decl) {
                variable_t &var = decl.variable();
                if (var.hasAttribute("shared")) {
                  var.add(0, threadgroup_q);
                }
              });
      }

      void metalParser::addBarriers() {
        statementArray::from(root)
            .flatFilterByStatementType(statementType::empty, "barrier")
            .forEach([&](statement_t *smnt) {
                // TODO 1.1: Implement proper barriers
                emptyStatement &emptySmnt = (emptyStatement&) *smnt;

                statement_t &barrierSmnt = (
                  *(new sourceCodeStatement(
                      emptySmnt.up,
                      emptySmnt.source,
                      "threadgroup_barrier(mem_flags::mem_threadgroup);"
                    ))
                );

                emptySmnt.replaceWith(barrierSmnt);

                delete &emptySmnt;
              });
      }

      void metalParser::setupHeaders() {
        strVector headers;
        headers.push_back("include <metal_stdlib>\n");
        headers.push_back("include <metal_compute>\n");

        const int headerCount = (int) headers.size();
        for (int i = 0; i < headerCount; ++i) {
          std::string header = headers[i];
          // TODO 1.1: Remove hack after methods are properly added
          if (i == 0) {
            header += "\nusing namespace metal;";
          }
          directiveToken token(root.source->origin,
                               header);
          root.addFirst(
            *(new directiveStatement(&root, token))
          );
        }
      }

      void metalParser::setupKernels() {
        root.children
            .filterByStatementType(
              statementType::functionDecl | statementType::function,
              "kernel"
            )
            .forEach([&](statement_t *smnt) {
                function_t *function;

                if (smnt->type() & statementType::functionDecl) {
                  function = &(((functionDeclStatement*) smnt)->function());

                  migrateLocalDecls((functionDeclStatement&) *smnt);
                  if (!success) return;
                } else {
                  function = &(((functionStatement*) smnt)->function());
                }

                setKernelQualifiers(*function);
            });
      }

      void metalParser::migrateLocalDecls(functionDeclStatement &kernelSmnt) {
        statementArray::from(kernelSmnt)
            .nestedForEachDeclaration([&](variableDeclaration &decl, declarationStatement &declSmnt) {
                variable_t &var = decl.variable();
                if (var.hasAttribute("shared")) {
                  declSmnt.removeFromParent();
                  kernelSmnt.addFirst(declSmnt);
                }
              });
      }

      void metalParser::setKernelQualifiers(function_t &function) {
        function.returnType.add(0, kernel_q);

        int argCount = (int) function.args.size();
        variablePtrVector constantArgs;
        for (int i = 0; i < argCount; ++i) {
          variable_t &arg = *(function.args[i]);
          arg.vartype = arg.vartype.flatten();
          if (arg.vartype.isPointerType()) {
            arg.add(0, device_q);
          } else {
            // - Set constant (replacing const)
            // - Pass the variable as a reference
            arg.add(0, constant_q);
            arg -= const_;
            arg.vartype.setReferenceToken(arg.source);
          }
          arg.vartype.customSuffix = "[[buffer(";
          arg.vartype.customSuffix += occa::toString(i);
          arg.vartype.customSuffix += ")]]";
        }

        variable_t occaGroupPositionArg(uint3, "_occa_group_position");
        variable_t occaThreadPositionArg(uint3, "_occa_thread_position");

        occaGroupPositionArg.vartype.customSuffix = (
          "[[threadgroup_position_in_grid]]"
        );
        occaThreadPositionArg.vartype.customSuffix = (
          "[[thread_position_in_threadgroup]]"
        );

        attribute_t &implicitArgAttr = *(getAttribute("implicitArg"));
        attributeToken_t groupAttr(implicitArgAttr, *(occaGroupPositionArg.source));
        attributeToken_t threadAttr(implicitArgAttr, *(occaThreadPositionArg.source));

        occaGroupPositionArg.addAttribute(groupAttr);
        occaThreadPositionArg.addAttribute(threadAttr);

        function.addArgument(occaGroupPositionArg);
        function.addArgument(occaThreadPositionArg);
      }
    }
  }
}
