#include <occa/internal/utils/string.hpp>
#include <occa/internal/lang/modes/opencl.hpp>
#include <occa/internal/lang/modes/okl.hpp>
#include <occa/internal/lang/modes/oklForStatement.hpp>
#include <occa/internal/lang/expr.hpp>
#include <occa/internal/lang/builtins/attributes.hpp>
#include <occa/internal/lang/builtins/types.hpp>

namespace occa {
  namespace lang {
    namespace okl {

      openclParser::openclParser(const occa::json &settings_) :
        withLauncher(settings_),
        constant("__constant", qualifierType::custom),
        kernel("__kernel", qualifierType::custom),
        local("__local", qualifierType::custom), 
        global("__global",qualifierType::custom) {

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

      std::string openclParser::launchBoundsAttribute(const int innerDims[3]) {
        std::stringstream ss; 
        ss << "__attribute__((reqd_work_group_size("
           << innerDims[0]
           << ","
           << innerDims[1]
           << ","
           << innerDims[2]
           << ")))\n";
        return ss.str();
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
        statementArray::from(root)
            .nestedForEachDeclaration([&](variableDeclaration &decl) {
                variable_t &var = decl.variable();
                if (var.hasAttribute("shared")) {
                  var.add(0,local);
                }
              });
      }

      void openclParser::setGlobalQualifiers() {
        statementArray::from(root)
            .nestedForEachDeclaration([&](variableDeclaration &decl) {
                variable_t &var = decl.variable();
                if (var.hasAttribute("globalPtr")) {
                  var.add(0,global);
                }
              });
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
        statementArray::from(root)
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
        statementArray::from(root)
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
        statementArray::from(root)
            .flatFilterByStatementType(
              statementType::blockStatements
              | statementType::function
            )
            .forEach([&](statement_t *smnt) {
               if (smnt->type() & statementType::function) {
                addStructToFunctionArgs(
                smnt->to<functionStatement>().function());
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
            });
      }

      void openclParser::setupKernels() {
        statementArray::from(root)
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

        for (auto arg : function.args) {
          vartype_t &type = arg->vartype;
          type = type.flatten();
          if (type.isPointerType())
            arg->add(0,global);
        }
      }

    }
  }
}
