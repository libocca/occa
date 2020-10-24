#include <occa/tools/string.hpp>
#include <occa/lang/modes/opencl.hpp>
#include <occa/lang/modes/okl.hpp>
#include <occa/lang/modes/oklForStatement.hpp>
#include <occa/lang/expr.hpp>
#include <occa/lang/builtins/attributes.hpp>
#include <occa/lang/builtins/types.hpp>

namespace occa {
  namespace lang {
    namespace okl {
      qualifier_t openclParser::global("__global", qualifierType::custom);

      openclParser::openclParser(const occa::properties &settings_) :
        withLauncher(settings_),
        constant("__constant", qualifierType::custom),
        kernel("__kernel", qualifierType::custom),
        local("__local", qualifierType::custom) {

        okl::addOklAttributes(*this);

        if (!settings.has("okl/restrict")) {
          settings["okl/restrict"] = "restrict";
        }
        settings["extensions/cl_khr_fp64"] = true;
      }

      void openclParser::onClear() {
        launcherClear();
      }

      void openclParser::beforePreprocessing() {
        preprocessor.addCompilerDefine("OCCA_USING_GPU", "1");
      }

      void openclParser::beforeKernelSplit() {
        if (!success) return;
        addExtensions();

        if (!success) return;
        updateConstToConstant();

        if (!success) return;
        setLocalQualifiers();

        if (!success) return;
        setGlobalQualifiers();
      }

      void openclParser::afterKernelSplit() {
        addBarriers();

        if (!success) return;
        addFunctionPrototypes();

        if (!success) return;
        addStructQualifiers();

        if (!success) return;
        setupKernels();
      }

      std::string openclParser::getOuterIterator(const int loopIndex) {
        std::string name = "get_group_id(";
        name += occa::toString(loopIndex);
        name += ')';
        return name;
      }

      std::string openclParser::getInnerIterator(const int loopIndex) {
        std::string name = "get_local_id(";
        name += occa::toString(loopIndex);
        name += ')';
        return name;
      }

      void openclParser::addExtensions() {
        if (!settings.has("extensions")) {
          return;
        }

        occa::json &extensions = settings["extensions"];
        if (!extensions.isObject()) {
          return;
        }

        jsonObject &extensionObj = extensions.object();
        jsonObject::iterator it = extensionObj.begin();
        while (it != extensionObj.end()) {
          const std::string &extension = it->first;
          const bool enabled = it->second;
          if (enabled) {
            root.addFirst(
              *(new pragmaStatement(
                  &root,
                  pragmaToken(root.source->origin,
                              "OPENCL EXTENSION "+ extension + " : enable\n")
                ))
            );
          }
          ++it;
        }
      }

      void openclParser::updateConstToConstant() {
        root.children
            .forEachDeclaration([&](variableDeclaration &decl) {
                variable_t &var = decl.variable();
                if (var.has(const_) && !var.has(typedef_)) {
                  var -= const_;
                  var.add(0, constant);
                }
              });
      }

      void openclParser::setLocalQualifiers() {
        root.children
            .nestedForEachDeclaration([&](variableDeclaration &decl) {
                variable_t &var = decl.variable();
                if (var.hasAttribute("shared")) {
                  var.add(0, local);
                }
              });
      }

      bool openclParser::sharedVariableMatcher(exprNode &expr) {
        return expr.hasAttribute("shared");
      }

      void openclParser::setGlobalQualifiers() {
        root.children
            .flatFilterByStatementType(
              statementType::declaration
              | statementType::functionDecl
              | statementType::function
            )
            .forEach(updateGlobalVariables);
      }

      void openclParser::updateGlobalVariables(statement_t *smnt) {
        if (smnt->type() & statementType::function) {
          addGlobalToFunctionArgs(
            smnt->to<functionStatement>().function()
          );
        }
        else if (smnt->type() & statementType::functionDecl) {
          addGlobalToFunctionArgs(
            smnt->to<functionDeclStatement>().function()
          );
        }
        else {
          declarationStatement &declSmnt = smnt->to<declarationStatement>();
          const int declCount = declSmnt.declarations.size();
          for (int i = 0; i < declCount; ++i) {
            addGlobalToVariable(
              declSmnt.declarations[i].variable()
            );
          }
        }
      }

      void openclParser::addGlobalToFunctionArgs(function_t &func) {
        const int argc = (int) func.args.size();
        for (int i = 0; i < argc; ++i) {
          variable_t *arg = func.args[i];
          if (arg) {
            addGlobalToVariable(*arg);
          }
        }
      }

      void openclParser::addGlobalToVariable(variable_t &var) {
        if (var.hasAttribute("globalPtr")) {
          var.add(0, global);
        }
      }

      void openclParser::updateScopeStructVariables(statement_t *smnt) {
        if (smnt->type() & statementType::function) {
          addStructToFunctionArgs(
            smnt->to<functionStatement>().function()
          );
          return;
        }

        scope_t &scope = smnt->to<blockStatement>().scope;

        keywordMap::iterator it = scope.keywords.begin();
        while (it != scope.keywords.end()) {
          keyword_t &keyword = *(it->second);

          if (keyword.type() & keywordType::variable) {
            addStructToVariable(keyword.to<variableKeyword>().variable);
          } else if (keyword.type() & keywordType::function) {
            addStructToFunctionArgs(
              keyword.to<functionKeyword>().function
            );
          }

          ++it;
        }
      }

      void openclParser::addStructToVariable(variable_t &var) {
        const type_t *type = var.vartype.type;
        if (type &&
            (type->type() & typeType::struct_) &&
            !var.has(struct_)) {
          var += struct_;
        }
      }

      void openclParser::addStructToFunctionArgs(function_t &func) {
        const int argc = (int) func.args.size();
        for (int i = 0; i < argc; ++i) {
          variable_t *arg = func.args[i];
          if (arg) {
            addStructToVariable(*arg);
          }
        }
      }

      void openclParser::addBarriers() {
        root.children
            .flatFilterByStatementType(statementType::empty, "barrier")
            .forEach([&](statement_t *smnt) {
                // TODO 1.1: Implement proper barriers
                emptyStatement &emptySmnt = (emptyStatement&) *smnt;

                statement_t &barrierSmnt = (
                  *(new sourceCodeStatement(
                      emptySmnt.up,
                      emptySmnt.source,
                      "barrier(CLK_LOCAL_MEM_FENCE);"
                    ))
                );

                emptySmnt.replaceWith(barrierSmnt);

                delete &emptySmnt;
              });
      }

      void openclParser::addFunctionPrototypes() {
        root.children
            .flatFilterByStatementType(statementType::functionDecl)
            .forEach([&](statement_t *smnt) {
                function_t &func = ((functionDeclStatement&) *smnt).function();
                functionStatement *funcSmnt = (
                  new functionStatement(&root,
                                        (function_t&) func.clone())
                );
                funcSmnt->attributes = smnt->attributes;

                root.addBefore(*smnt, *funcSmnt);
              });
      }

      void openclParser::addStructQualifiers() {
        root.children
            .flatFilterByStatementType(
              statementType::blockStatements
              | statementType::function
            )
            .forEach(updateScopeStructVariables);
      }

      void openclParser::setupKernels() {
        root.children
            .flatFilterByStatementType(
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

      void openclParser::migrateLocalDecls(functionDeclStatement &kernelSmnt) {
        statementArray::from(kernelSmnt)
            .nestedForEachDeclaration([&](variableDeclaration &decl, declarationStatement &declSmnt) {
                variable_t &var = decl.variable();
                if (var.hasAttribute("shared")) {
                  declSmnt.removeFromParent();
                  kernelSmnt.addFirst(declSmnt);
                }
              });
      }

      void openclParser::setKernelQualifiers(function_t &function) {
        function.returnType.add(0, kernel);

        const int argCount = (int) function.args.size();
        for (int ai = 0; ai < argCount; ++ai) {
          variable_t &arg = *(function.args[ai]);
          arg.vartype = arg.vartype.flatten();
          if (arg.vartype.isPointerType()) {
            arg.add(0, global);
          }
        }
      }
    }
  }
}
