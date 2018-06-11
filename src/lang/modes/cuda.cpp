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
#include <occa/lang/modes/cuda.hpp>
#include <occa/lang/modes/okl.hpp>
#include <occa/lang/modes/oklForStatement.hpp>
#include <occa/lang/builtins/attributes.hpp>
#include <occa/lang/builtins/types.hpp>

/*
//---[ Loop Info ]--------------------------------
#define occaOuterDim2 gridDim.z
#define occaOuterId2  blockIdx.z

#define occaOuterDim1 gridDim.y
#define occaOuterId1  blockIdx.y

#define occaOuterDim0 gridDim.x
#define occaOuterId0  blockIdx.x
// - - - - - - - - - - - - - - - - - - - - - - - -
#define occaInnerDim2 blockDim.z
#define occaInnerId2  threadIdx.z

#define occaInnerDim1 blockDim.y
#define occaInnerId1  threadIdx.y

#define occaInnerDim0 blockDim.x
#define occaInnerId0  threadIdx.x
//================================================


//---[ Standard Functions ]-----------------------
@barrier("local")  __syncthreads()
@barrier("global") __syncthreads()
//================================================


//---[ Attributes ]-------------------------------
#define occaShared   __shared__
#define occaRestrict __restrict__
#define occaConstant __constant__
//================================================


//---[ Kernel Info ]------------------------------
#define occaKernel         extern "C" __global__
#define occaFunction       __device__
#define occaDeviceFunction __device__
//================================================
 */

namespace occa {
  namespace lang {
    namespace okl {
      cudaParser::cudaParser(const occa::properties &settings_) :
        withLauncher(settings_),
        restrict_("__restrict__", (qualifierType::forPointers_ |
                                   qualifierType::custom)),
        constant("__constant__", qualifierType::custom),
        global("__global__", qualifierType::custom),
        device("__device__", qualifierType::custom),
        shared("__shared__", qualifierType::custom) {

        addAttribute<attributes::kernel>();
        addAttribute<attributes::outer>();
        addAttribute<attributes::inner>();
        addAttribute<attributes::shared>();
        addAttribute<attributes::exclusive>();

        if (settings.has("cuda/restrict")) {
          occa::json &r = settings["cuda/restrict"];
          if (r.isString()) {
            restrict_.name = r.string();
          } else if (r.isBoolean()
                     && !r.boolean()) {
            restrict_.name = "";
          }
        }

        if (restrict_.name.size()) {
          replaceKeyword(keywords,
                         new qualifierKeyword(restrict_));
        }
      }

      void cudaParser::onClear() {
        hostClear();
      }

      void cudaParser::beforePreprocessing() {
        preprocessor.addCompilerDefine("restrict",
                                       restrict_.name);
      }

      void cudaParser::beforeKernelSplit() {
        if (!success) return;
        updateConstToConstant();

        if (!success) return;
        setFunctionQualifiers();

        if (!success) return;
        setSharedQualifiers();
      }

      void cudaParser::afterKernelSplit() {
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

      void cudaParser::setFunctionQualifiers() {
        statementPtrVector funcSmnts;
        findStatementsByType(statementType::functionDecl,
                             root,
                             funcSmnts);

        const int funcCount = (int) funcSmnts.size();
        for (int i = 0; i < funcCount; ++i) {
          functionDeclStatement &funcSmnt = (
            *((functionDeclStatement*) funcSmnts[i])
          );
          if (funcSmnt.hasAttribute("kernel")) {
            continue;
          }
          vartype_t &vartype = funcSmnt.function.returnType;
          vartype.qualifiers.addFirst(vartype.origin(),
                                      device);
        }
      }

      void cudaParser::setSharedQualifiers() {
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
            var += shared;
          }
          ++it;
        }
      }

      void cudaParser::setupKernels() {
        statementPtrVector kernelSmnts;
        findStatementsByAttr(statementType::functionDecl,
                             "kernel",
                             root,
                             kernelSmnts);

        const int kernelCount = (int) kernelSmnts.size();
        for (int i = 0; i < kernelCount; ++i) {
          functionDeclStatement &kernelSmnt = (
            *((functionDeclStatement*) kernelSmnts[i])
          );
          setKernelQualifiers(kernelSmnt);
          if (!success) return;
        }
      }

      void cudaParser::setKernelQualifiers(functionDeclStatement &kernelSmnt) {
        vartype_t &vartype = kernelSmnt.function.returnType;
        vartype.qualifiers.addFirst(vartype.origin(),
                                    global);
        vartype.qualifiers.addFirst(vartype.origin(),
                                    externC);
      }

      bool cudaParser::sharedVariableMatcher(exprNode &expr) {
        return expr.hasAttribute("shared");
      }
    }
  }
}
