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
      openclParser::openclParser(const occa::properties &settings_) :
        parser_t(settings_),
        hostParser(settings["host"]),
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

      //---[ Public ]-------------------
      void openclParser::writeHostSourceToFile(const std::string &filename) const {
        hostParser.writeToFile(filename);
      }
      //================================

      void openclParser::onClear() {
        hostParser.onClear();
      }

      void openclParser::afterParsing() {
        if (!success) return;
        if (settings.get("okl/validate", true)) {
          checkKernels(root);
        }

        if (!success) return;
        setupHostParser();

        if (!success) return;
        addExtensions();

        if (!success) return;
        updateConstToConstant();

        if (!success) return;
        setLocalQualifiers();

        if (!success) return;
        setupKernels();

        if (!success) return;
        addFunctionPrototypes();
      }

      void openclParser::setupHostParser() {
        // Clone source
        blockStatement &rootClone = (blockStatement&) root.clone();
        hostParser.root.swap(rootClone);
        delete &rootClone;
        hostParser.setupKernels();

        // Remove outer loops
        statementPtrVector kernelSmnts;
        findStatementsByAttr(statementType::functionDecl,
                             "kernel",
                             hostParser.root,
                             kernelSmnts);

        const int kernelCount = (int) kernelSmnts.size();
        for (int i = 0; i < kernelCount; ++i) {
          functionDeclStatement &kernelSmnt = (
            *((functionDeclStatement*) kernelSmnts[i])
          );
          removeHostOuterLoops(kernelSmnt);
          if (!success) {
            return;
          }
        }

        setupHostHeaders();
      }

      void openclParser::setupHostHeaders() {
        // TODO 1.1: Remove hack after methods are properly added
        const int headerCount = 2;
        std::string headers[headerCount] = {
          "include <occa/base.hpp>",
          "include <occa/modes/serial/kernel.hpp>"
        };
        for (int i = 0; i < headerCount; ++i) {
          std::string header = headers[i];
          directiveToken token(root.source->origin,
                               header);
          hostParser.root.addFirst(
            *(new directiveStatement(&root, token))
          );
        }
      }

      void openclParser::removeHostOuterLoops(functionDeclStatement &kernelSmnt) {
        statementPtrVector outerSmnts;
        findStatementsByAttr(statementType::for_,
                             "outer",
                             kernelSmnt,
                             outerSmnts);

        const int outerCount = (int) outerSmnts.size();
        for (int i = 0; i < outerCount; ++i) {
          forStatement &forSmnt = *((forStatement*) outerSmnts[i]);
          if (!isOuterMostOuterLoop(forSmnt)) {
            continue;
          }
          setKernelLaunch(kernelSmnt, forSmnt);
        }
      }

      bool openclParser::isOuterMostOuterLoop(forStatement &forSmnt) {
        statement_t *smnt = forSmnt.up;
        while (smnt) {
          if ((smnt->type() & statementType::for_)
              && smnt->hasAttribute("outer")) {
            return false;
          }
          smnt = smnt->up;
        }
        return true;
      }

      void openclParser::setKernelLaunch(functionDeclStatement &kernelSmnt,
                                         forStatement &forSmnt) {
        forStatement *innerSmnt = getInnerMostInnerLoop(forSmnt);
        if (!innerSmnt) {
          success = false;
          forSmnt.printError("No [@inner] for-loop found");
          return;
        }

        statementPtrVector path;
        oklForStatement::getOKLLoopPath(*innerSmnt, path);

        // Create block in case there are duplicate variable names
        blockStatement &launchBlock = (
          *new blockStatement(forSmnt.up, forSmnt.source)
        );
        forSmnt.up->addBefore(forSmnt, launchBlock);

        int outerCount = 0;
        int innerCount = 0;

        // **TODO 1.1: Properly fix this
        const int pathCount = (int) path.size();
        for (int i = 0; i < pathCount; ++i) {
          forStatement &pathSmnt = *((forStatement*) path[i]);
          oklForStatement oklForSmnt(pathSmnt);
          if (!oklForSmnt.isValid()) {
            success = false;
            return;
          }

          const bool isOuter = pathSmnt.hasAttribute("outer");
          const int index = (isOuter
                             ? outerCount
                             : innerCount);
          token_t *source = pathSmnt.source;
          const std::string &name = (isOuter
                                     ? "outer"
                                     : "inner");
          launchBlock.add(
            *(new expressionStatement(
                &launchBlock,
                setDim(source, name, index,
                       oklForSmnt.getIterationCount())
              ))
          );

          outerCount += isOuter;
          innerCount += !isOuter;
        }

        launchBlock.addFirst(
          *(new expressionStatement(
              &launchBlock,
              *(new identifierNode(forSmnt.source,
                                   "inner.dims = " + occa::toString(innerCount)))
            ))
        );
        launchBlock.addFirst(
          *(new expressionStatement(
              &launchBlock,
              *(new identifierNode(forSmnt.source,
                                   "outer.dims = " + occa::toString(outerCount)))
            ))
        );
        launchBlock.addFirst(
          *(new expressionStatement(
              &launchBlock,
              *(new identifierNode(forSmnt.source,
                                   "occa::dim outer, inner"))
            ))
        );
        // Wrap kernel
        launchBlock.add(
          *(new expressionStatement(
              &launchBlock,
              *(new identifierNode(forSmnt.source,
                                   "occa::kernel kernel(deviceKernel)"))
            ))
        );
        // Set run dims
        launchBlock.add(
          *(new expressionStatement(
              &launchBlock,
              *(new identifierNode(forSmnt.source,
                                   "kernel.setRunDims(outer, inner)"))
            ))
        );
        // launch kernel
        std::string kernelCall = "kernel(";
        function_t &func = kernelSmnt.function;
        const int argCount = (int) func.args.size();
        for (int i = 0; i < argCount; ++i) {
          if (i) {
            kernelCall += ", ";
          }
          kernelCall += func.args[i]->name();
        }
        kernelCall += ')';
        launchBlock.add(
          *(new expressionStatement(
              &launchBlock,
              *(new identifierNode(forSmnt.source,
                                   kernelCall))
            ))
        );
        // Add kernel argument
        identifierToken kernelTypeSource(forSmnt.source->origin,
                                         "occa::kernel_v");
        type_t &kernelType = *(new typedef_t(vartype_t(),
                                             kernelTypeSource));
        identifierToken kernelVarSource(forSmnt.source->origin,
                                        "*deviceKernel");
        variable_t &kernelVar = *(new variable_t(kernelType,
                                                 &kernelVarSource));
        func.args.insert(func.args.begin(),
                         &kernelVar);

        kernelSmnt.scope.add(kernelType);
        kernelSmnt.scope.add(kernelVar);

        forSmnt.removeFromParent();
        delete &forSmnt;
      }

      int openclParser::getInnerLoopLevel(forStatement &forSmnt) {
        statement_t *smnt = forSmnt.up;
        int level = 0;
        while (smnt) {
          if ((smnt->type() & statementType::for_)
              && smnt->hasAttribute("inner")) {
            ++level;
          }
          smnt = smnt->up;
        }
        return level;
      }

      forStatement* openclParser::getInnerMostInnerLoop(forStatement &forSmnt) {
        statementPtrVector innerSmnts;
        findStatementsByAttr(statementType::for_,
                             "inner",
                             forSmnt,
                             innerSmnts);

        int maxLevel = -1;
        forStatement *innerMostInnerLoop = NULL;

        const int innerCount = (int) innerSmnts.size();
        for (int i = 0; i < innerCount; ++i) {
          forStatement &innerSmnt = *((forStatement*) innerSmnts[i]);
          const int level = getInnerLoopLevel(innerSmnt);
          if (level > maxLevel) {
            maxLevel = level;
            innerMostInnerLoop = &innerSmnt;
          }
        }

        return innerMostInnerLoop;
      }

      exprNode& openclParser::setDim(token_t *source,
                                     const std::string &name,
                                     const int index,
                                     exprNode *value) {
        identifierNode var(source, name);
        primitiveNode idx(source, index);
        subscriptNode access(source, var, idx);
        exprNode &assign = (
          *(new binaryOpNode(source,
                             op::assign,
                             access,
                             *value))
        );
        delete value;
        return assign;
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

      void openclParser::setupKernels() {
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

          replaceOccaFors(kernelSmnt);
          if (!success) return;
        }
      }

      void openclParser::setKernelQualifiers(functionDeclStatement &kernelSmnt) {
        function_t &func = kernelSmnt.function;
        func.returnType += kernel;

        const int argCount = (int) func.args.size();
        for (int ai = 0; ai < argCount; ++ai) {
          variable_t &arg = *(func.args[ai]);
          if (arg.vartype.isPointerType()) {
            arg += global;
          }
        }
      }

      void openclParser::replaceOccaFors(functionDeclStatement &kernelSmnt) {
        statementPtrVector outerSmnts, innerSmnts;
        findStatementsByAttr(statementType::for_,
                             "outer",
                             kernelSmnt,
                             outerSmnts);
        findStatementsByAttr(statementType::for_,
                             "inner",
                             kernelSmnt,
                             innerSmnts);

        const int outerCount = (int) outerSmnts.size();
        for (int i = 0; i < outerCount; ++i) {
          replaceOccaFor(*((forStatement*) outerSmnts[i]));
        }

        const int innerCount = (int) innerSmnts.size();
        for (int i = 0; i < innerCount; ++i) {
          replaceOccaFor(*((forStatement*) innerSmnts[i]));
        }
      }


      void openclParser::replaceOccaFor(forStatement &forSmnt) {
        oklForStatement oklForSmnt(forSmnt);

        // Replace for-loops with blocks
        const int childIndex = forSmnt.childIndex();
        blockStatement &blockSmnt = *(new blockStatement(forSmnt.up,
                                                         forSmnt.source));
        blockSmnt.swap(forSmnt);
        blockSmnt.up->children[childIndex] = &blockSmnt;

        // Get OpenCL iterator
        std::string clIteratorName;
        if (oklForSmnt.isOuterLoop()) {
          clIteratorName = "get_group_id(";
        } else {
          clIteratorName = "get_local_id(";
        }
        clIteratorName += occa::toString(oklForSmnt.oklLoopIndex());
        clIteratorName += ')';

        identifierToken clIteratorSource(oklForSmnt.iterator->source->origin,
                                         clIteratorName);
        identifierNode clIterator(&clIteratorSource,
                                  clIteratorName);

        // Create iterator declaration
        variableDeclaration decl;
        decl.variable = oklForSmnt.iterator;
        decl.value = oklForSmnt.makeDeclarationValue(clIterator);

        // Add declaration before block
        declarationStatement &declSmnt = (
          *(new declarationStatement(blockSmnt.up))
        );
        declSmnt.declarations.push_back(decl);

        blockSmnt.addFirst(declSmnt);
        delete &forSmnt;
      }

      bool openclParser::sharedVariableMatcher(exprNode &expr) {
        return expr.hasAttribute("shared");
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
