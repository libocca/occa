#include <occa/tools/string.hpp>
#include <occa/lang/modes/dpcpp.hpp>
#include <occa/lang/modes/okl.hpp>
#include <occa/lang/modes/oklForStatement.hpp>
#include <occa/lang/builtins/attributes.hpp>
#include <occa/lang/builtins/types.hpp>

namespace occa {
  namespace lang {
    namespace okl {
      qualifier_t dpcppParser::global("__global", qualifierType::custom);

      dpcppParser::dpcppParser(const occa::properties &settings_) :
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

      void dpcppParser::onClear() {
        launcherClear();
      }

      void dpcppParser::beforePreprocessing() {
        preprocessor.addCompilerDefine("OCCA_USING_GPU", "1");
      }

      void dpcppParser::beforeKernelSplit() {
        if (!success) return;
        addExtensions();

        if (!success) return;
        updateConstToConstant();

        if (!success) return;
        setLocalQualifiers();

        if (!success) return;
        setGlobalQualifiers();
      }

      void dpcppParser::afterKernelSplit() {
        addBarriers();

        if (!success) return;
        addFunctionPrototypes();

        if (!success) return;
        addStructQualifiers();

        if (!success) return;
        setupKernels();
      }

      std::string dpcppParser::getOuterIterator(const int loopIndex) {
        std::string name = "get_group_id(";
        name += occa::toString(loopIndex);
        name += ')';
        return name;
      }

      std::string dpcppParser::getInnerIterator(const int loopIndex) {
        std::string name = "get_local_id(";
        name += occa::toString(loopIndex);
        name += ')';
        return name;
      }

      void dpcppParser::addExtensions() {
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

      void dpcppParser::updateConstToConstant() {
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

      void dpcppParser::setLocalQualifiers() {
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

      bool dpcppParser::sharedVariableMatcher(exprNode &expr) {
        return expr.hasAttribute("shared");
      }

      void dpcppParser::setGlobalQualifiers() {
        statementPtrVector statements;
        findStatements((statementType::declaration |
                        statementType::functionDecl |
                        statementType::function),
                       root,
                       updateGlobalVariables,
                       statements);
      }

      bool dpcppParser::updateGlobalVariables(statement_t &smnt) {
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

      void dpcppParser::addGlobalToFunctionArgs(function_t &func) {
        const int argc = (int) func.args.size();
        for (int i = 0; i < argc; ++i) {
          variable_t *arg = func.args[i];
          if (arg) {
            addGlobalToVariable(*arg);
          }
        }
      }

      void dpcppParser::addGlobalToVariable(variable_t &var) {
        if (var.hasAttribute("globalPtr")) {
          var.add(0, global);
        }
      }

      bool dpcppParser::updateScopeStructVariables(statement_t &smnt) {
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

      void dpcppParser::addStructToVariable(variable_t &var) {
        const type_t *type = var.vartype.type;
        if (type &&
            (type->type() & typeType::struct_) &&
            !var.has(struct_)) {
          var += struct_;
        }
      }

      void dpcppParser::addStructToFunctionArgs(function_t &func) {
        const int argc = (int) func.args.size();
        for (int i = 0; i < argc; ++i) {
          variable_t *arg = func.args[i];
          if (arg) {
            addStructToVariable(*arg);
          }
        }
      }

      void dpcppParser::addBarriers() {
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

      void dpcppParser::addFunctionPrototypes() {
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

      void dpcppParser::addStructQualifiers() {
        statementPtrVector statements;
        findStatements(statementType::blockStatements |
                       statementType::function,
                       root,
                       updateScopeStructVariables,
                       statements);
      }

      void dpcppParser::setupKernels() {
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

      void dpcppParser::migrateLocalDecls(functionDeclStatement &kernelSmnt) {
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

      void dpcppParser::setKernelQualifiers(function_t &function) {
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
