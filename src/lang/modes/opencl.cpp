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
#include "occa/lang/modes/opencl.hpp"
#include "occa/lang/modes/okl.hpp"
#include "occa/lang/builtins/attributes.hpp"
#include "occa/lang/builtins/types.hpp"

/*
//---[ Loop Info ]--------------------------------
#define occaOuterDim2 (get_num_groups(2))
#define occaOuterId2  (get_group_id(2))

#define occaOuterDim1 (get_num_groups(1))
#define occaOuterId1  (get_group_id(1))

#define occaOuterDim0 (get_num_groups(0))
#define occaOuterId0  (get_group_id(0))
// - - - - - - - - - - - - - - - - - - - - - - - -
#define occaInnerDim2 (get_local_size(2))
#define occaInnerId2  (get_local_id(2))

#define occaInnerDim1 (get_local_size(1))
#define occaInnerId1  (get_local_id(1))

#define occaInnerDim0 (get_local_size(0))
#define occaInnerId0  (get_local_id(0))
//================================================


//---[ Standard Functions ]-----------------------
@barrier("local")  -> barrier(CLK_LOCAL_MEM_FENCE)
@barrier("global") -> barrier(CLK_GLOBAL_MEM_FENCE)
//================================================


//---[ Attributes ]-------------------------------
#define occaShared   __local
#define occaPointer  __global
#define occaConstant __constant
//================================================


//---[ Kernel Info ]------------------------------
#define occaKernel         __kernel
#define occaFunction
#define occaDeviceFunction
//================================================
 */

namespace occa {
  namespace lang {
    namespace okl {
      openclParser::openclParser() :
        hostParser(settings),
        constant("__constant", qualifierType::custom),
        kernel("__kernel", qualifierType::custom),
        global("__global", qualifierType::custom),
        local("__local", qualifierType::custom) {

        addAttribute<attributes::kernel>();
        addAttribute<attributes::outer>();
        addAttribute<attributes::inner>();
        addAttribute<attributes::shared>();
        addAttribute<attributes::exclusive>();

        settings["opencl/extensions/cl_khr_fp64"] = true;
        hostParser.settings["okl/validate"] = false;
      }

      void openclParser::onClear() {
        hostParser.onClear();
      }

      void openclParser::afterParsing() {
        if (!success) return;
        if (settings.get("okl/validate", true)) {
          checkKernels(root);
        }

        blockStatement &rootClone = (blockStatement&) root.clone();
        hostParser.root.swap(rootClone);
        delete &rootClone;
        hostParser.setupKernels();
        std::cout << "hostParser.toString() = \n"
                  << "--------------------------------------------------\n"
                  << hostParser.toString()
                  << "==================================================\n";

        if (!success) return;
        addExtensions();

        if (!success) return;
        setQualifiers();

        if (!success) return;
        addOccaFors();

        if (!success) return;
        setupLaunchKernel();

        if (!success) return;
        addFunctionPrototypes();
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
                              "OPENCL EXTENSION "+ extension + " : enable")
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

      void openclParser::setQualifiers() {
        updateConstToConstant();

        if (!success) return;
        setLocalQualifiers();

        if (!success) return;
        setKernelQualifiers();
      }

      void openclParser::setKernelQualifiers() {
        statementPtrVector kernelSmnts;
        findStatementsByAttr(statementType::functionDecl,
                             "kernel",
                             root,
                             kernelSmnts);
        const int kernelCount = (int) kernelSmnts.size();
        for (int i = 0; i < kernelCount; ++i) {
          function_t &func = ((functionDeclStatement*) kernelSmnts[i])->function;
          func.returnType += kernel;

          const int argCount = (int) func.args.size();
          for (int ai = 0; ai < argCount; ++ai) {
            variable_t &arg = *(func.args[ai]);
            if (arg.vartype.isPointerType()) {
              arg += global;
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

      void openclParser::addOccaFors() {
      }

      void openclParser::setupLaunchKernel() {
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
          root.add(*funcSmnt, index - 1);
          ++index;
        }
      }
    }
  }
}
