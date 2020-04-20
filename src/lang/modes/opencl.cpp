#include <occa/tools/string.hpp>
#include <occa/lang/modes/opencl.hpp>
#include <occa/lang/modes/okl.hpp>
#include <occa/lang/modes/oklForStatement.hpp>
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

        okl::addAttributes(*this);

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
        const int childCount = (int) root.children.size();
        for (int i = 0; i < childCount; ++i) {
          statement_t &child = *(root.children[i]);
          if (child.type() != statementType::declaration) {
            continue;
          }
          declarationStatement &declSmnt = ((declarationStatement&) child);
          const int declCount = declSmnt.declarations.size();
          for (int di = 0; di < declCount; ++di) {
            variable_t &var = *(declSmnt.declarations[di].variable);
            if (var.has(const_) && !var.has(typedef_)) {
              var -= const_;
              var.add(0, constant);
            }
          }
        }
      }

      void openclParser::setLocalQualifiers() {
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
            var.add(0, local);
          }
          ++it;
        }
      }

      bool openclParser::sharedVariableMatcher(exprNode &expr) {
        return expr.hasAttribute("shared");
      }

      void openclParser::setGlobalQualifiers() {
        statementPtrVector statements;
        findStatements((statementType::declaration |
                        statementType::struct_ |
                        statementType::functionDecl |
                        statementType::function),
                       root,
                       updateGlobalVariables,
                       statements);
      }

      bool openclParser::updateGlobalVariables(statement_t &smnt) {
        if (smnt.type() & statementType::function) {
          addGlobalToFunctionArgs(
            smnt.to<functionStatement>().function
          );
        }
        else if (smnt.type() & statementType::functionDecl) {
          addGlobalToFunctionArgs(
            smnt.to<functionDeclStatement>().function
          );
        }
        else if (smnt.type() & statementType::struct_) {
          addGlobalToStruct(
            smnt.to<structStatement>().struct_
          );
        }
        else {
          declarationStatement &declSmnt = smnt.to<declarationStatement>();
          const int declCount = declSmnt.declarations.size();
          for (int i = 0; i < declCount; ++i) {
            addGlobalToVariable(
              *(declSmnt.declarations[i].variable)
            );
          }
        }
        return false;
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

      void openclParser::addGlobalToStruct(struct_t &struct__) {
        const int fieldCount = (int) struct__.fields.size();
        for (int i = 0; i < fieldCount; ++i) {
          addGlobalToVariable(struct__.fields[i]);
        }
      }

      void openclParser::addGlobalToVariable(variable_t &var) {
        if (var.hasAttribute("globalPtr")) {
          var.add(0, global);
        }
      }

      bool openclParser::updateScopeStructVariables(statement_t &smnt) {
        if (smnt.type() & statementType::function) {
          addStructToFunctionArgs(
            smnt.to<functionStatement>().function
          );
          return false;
        }

        scope_t &scope = smnt.to<blockStatement>().scope;

        keywordMap::iterator it = scope.keywords.begin();
        while (it != scope.keywords.end()) {
          keyword_t &keyword = *(it->second);

          if (keyword.type() & keywordType::variable) {
            addStructToVariable(keyword.to<variableKeyword>().variable);
          } else if (keyword.type() & keywordType::function) {
            addStructToFunctionArgs(keyword.to<functionKeyword>().function);
          }

          ++it;
        }

        return false;
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
                                     "barrier(CLK_LOCAL_MEM_FENCE)"))
              ))
          );

          smnt.up->addBefore(smnt,
                             barrierSmnt);

          smnt.up->remove(smnt);
          delete &smnt;
        }
      }

      void openclParser::addFunctionPrototypes() {
        const int childCount = (int) root.children.size();
        int index = 0;
        for (int i = 0; i < childCount; ++i) {
          statement_t &child = *(root.children[index]);
          ++index;
          if (child.type() != statementType::functionDecl) {
            continue;
          }
          function_t &func = ((functionDeclStatement&) child).function;
          functionStatement *funcSmnt = (
            new functionStatement(&root,
                                  (function_t&) func.clone())
          );
          funcSmnt->attributes = child.attributes;

          root.add(*funcSmnt, index - 1);
          ++index;
        }
      }

      void openclParser::addStructQualifiers() {
        statementPtrVector statements;
        findStatements(statementType::blockStatements |
                       statementType::function,
                       root,
                       updateScopeStructVariables,
                       statements);
      }

      void openclParser::setupKernels() {
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

      void openclParser::migrateLocalDecls(functionDeclStatement &kernelSmnt) {
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
