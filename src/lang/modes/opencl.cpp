/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 David Medina and Tim Warburton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */
#include <occa/tools/string.hpp>
#include <occa/lang/modes/opencl.hpp>
#include <occa/lang/modes/okl.hpp>
#include <occa/lang/modes/oklForStatement.hpp>
#include <occa/lang/builtins/attributes.hpp>
#include <occa/lang/builtins/types.hpp>

namespace occa {
  namespace lang {
    namespace okl {
      openclParser::openclParser(const occa::properties &settings_) :
        withLauncher(settings_),
        constant("__constant", qualifierType::custom),
        kernel("__kernel", qualifierType::custom),
        global("__global", qualifierType::custom),
        local("__local", qualifierType::custom) {

        okl::addAttributes(*this);

        settings["opencl/extensions/cl_khr_fp64"] = true;
      }

      void openclParser::onClear() {
        hostClear();
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
      }

      void openclParser::afterKernelSplit() {
        addBarriers();

        if (!success) return;
        addFunctionPrototypes();

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
        if (!settings.has("opencl/extensions")) {
          return;
        }

        occa::json &extensions = settings["opencl/extensions"];
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
            if (var.has(const_)) {
              var -= const_;
              var += constant;
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
            var += local;
          }
          ++it;
        }
      }

      bool openclParser::sharedVariableMatcher(exprNode &expr) {
        return expr.hasAttribute("shared");
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
          } else {
            function = &(((functionStatement*) kernelSmnts[i])->function);
          }
          setKernelQualifiers(*function);
          if (!success) return;
        }
      }

      void openclParser::setKernelQualifiers(function_t &function) {
        function.returnType += kernel;

        const int argCount = (int) function.args.size();
        for (int ai = 0; ai < argCount; ++ai) {
          variable_t &arg = *(function.args[ai]);
          arg.vartype = arg.vartype.flatten();
          if (arg.vartype.isPointerType()) {
            arg += global;
          }
        }
      }
    }
  }
}
