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
#include <occa/lang/modes/withLauncher.hpp>
#include <occa/lang/modes/okl.hpp>
#include <occa/lang/modes/oklForStatement.hpp>
#include <occa/lang/builtins/attributes.hpp>
#include <occa/lang/builtins/types.hpp>
#include <occa/lang/builtins/transforms/replacer.hpp>

namespace occa {
  namespace lang {
    namespace okl {
      withLauncher::withLauncher(const occa::properties &settings_) :
        parser_t(settings_),
        hostParser(settings["host"]) {
        hostParser.settings["okl/validate"] = false;
      }

      //---[ Public ]-------------------
      void withLauncher::writeHostSourceToFile(const std::string &filename) const {
        hostParser.writeToFile(filename);
      }
      //================================

      void withLauncher::hostClear() {
        hostParser.onClear();
      }

      void withLauncher::afterParsing() {
        if (!success) return;
        if (settings.get("okl/validate", true)) {
          checkKernels(root);
        }

        if (!success) return;
        setOKLLoopIndices();

        if (!success) return;
        setupHostParser();

        if (!success) return;
        beforeKernelSplit();

        if (!success) return;
        splitKernels();

        if (!success) return;
        setupKernels();

        if (!success) return;
        afterKernelSplit();
      }

      void withLauncher::beforeKernelSplit() {}

      void withLauncher::afterKernelSplit() {}

      void withLauncher::setOKLLoopIndices() {
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
          okl::setLoopIndices(kernelSmnt);
          if (!success) return;
        }
      }

      void withLauncher::setupHostParser() {
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
          if (!success) return;
          setupHostKernelArgs(kernelSmnt);
          if (!success) return;
        }

        setupHostHeaders();
      }

      void withLauncher::removeHostOuterLoops(functionDeclStatement &kernelSmnt) {
        statementPtrVector outerSmnts;
        findStatementsByAttr(statementType::for_,
                             "outer",
                             kernelSmnt,
                             outerSmnts);

        const int outerCount = (int) outerSmnts.size();
        int kernelIndex = 0;
        for (int i = 0; i < outerCount; ++i) {
          forStatement &forSmnt = *((forStatement*) outerSmnts[i]);
          if (!isOuterMostOuterLoop(forSmnt)) {
            continue;
          }
          setKernelLaunch(kernelSmnt,
                          forSmnt,
                          kernelIndex++);
        }
      }

      bool withLauncher::isOuterMostOuterLoop(forStatement &forSmnt) {
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

      void withLauncher::setKernelLaunch(functionDeclStatement &kernelSmnt,
                                         forStatement &forSmnt,
                                         const int kernelIndex) {
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

          launchBlock.add(pathSmnt.init->clone(&launchBlock));

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
        std::stringstream ss;
        ss << "occa::kernel kernel(deviceKernel["
           << kernelIndex
           << "])";
        launchBlock.add(
          *(new expressionStatement(
              &launchBlock,
              *(new identifierNode(forSmnt.source,
                                   ss.str()))
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

        forSmnt.removeFromParent();

        // TODO 1.1: Delete after properly cloning the declaration statement
        // delete &forSmnt;
      }

      void withLauncher::setupHostKernelArgs(functionDeclStatement &kernelSmnt) {
        // Add kernel argument
        identifierToken kernelTypeSource(kernelSmnt.source->origin,
                                         "occa::kernel_v");
        type_t &kernelType = *(new typedef_t(vartype_t(),
                                             kernelTypeSource));
        identifierToken kernelVarSource(kernelSmnt.source->origin,
                                        "*deviceKernel");
        variable_t &kernelVar = *(new variable_t(kernelType,
                                                 &kernelVarSource));
        kernelVar += pointer_t();

        function_t &func = kernelSmnt.function;
        func.args.insert(func.args.begin(),
                         &kernelVar);

        kernelSmnt.scope.add(kernelType);
        kernelSmnt.scope.add(kernelVar);
      }

      void withLauncher::setupHostHeaders() {
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

      int withLauncher::getInnerLoopLevel(forStatement &forSmnt) {
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

      forStatement* withLauncher::getInnerMostInnerLoop(forStatement &forSmnt) {
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

      exprNode& withLauncher::setDim(token_t *source,
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

      void withLauncher::splitKernels() {
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
          splitKernel(kernelSmnt);
          if (!success) return;
        }
      }

      void withLauncher::splitKernel(functionDeclStatement &kernelSmnt) {
        statementPtrVector outerSmnts;
        findStatementsByAttr(statementType::for_,
                             "outer",
                             kernelSmnt,
                             outerSmnts);

        statementPtrVector newKernelSmnts;

        const int outerCount = (int) outerSmnts.size();
        int kernelIndex = 0;
        for (int i = 0; i < outerCount; ++i) {
          forStatement &forSmnt = *((forStatement*) outerSmnts[i]);
          if (!isOuterMostOuterLoop(forSmnt)) {
            continue;
          }
          newKernelSmnts.push_back(
            extractLoopAsKernel(kernelSmnt,
                                forSmnt,
                                kernelIndex++)
          );
        }

        int smntIndex = kernelSmnt.childIndex();
        for (int i = (kernelIndex - 1); i >= 0; --i) {
          root.add(*(newKernelSmnts[i]),
                   smntIndex);
        }

        root.remove(kernelSmnt);
        root.scope.remove(kernelSmnt.function.name(),
                          true);

        // TODO 1.1: Find out what causes segfault here
        // delete &kernelSmnt;
      }

      statement_t* withLauncher::extractLoopAsKernel(functionDeclStatement &kernelSmnt,
                                                     forStatement &forSmnt,
                                                     const int kernelIndex) {

        function_t &oldFunction = kernelSmnt.function;
        function_t &newFunction = (function_t&) oldFunction.clone();
        std::stringstream ss;
        ss << +"_occa_" << newFunction.name() << "_" << kernelIndex;
        newFunction.source->value = ss.str();

        functionDeclStatement &newKernelSmnt = *(
          new functionDeclStatement(&root,
                                    newFunction)
        );
        newKernelSmnt.attributes = kernelSmnt.attributes;

        // Clone for-loop and replace argument variables
        statement_t &newForSmnt = forSmnt.clone();
        const int argc = (int) newFunction.args.size();
        for (int i = 0; i < argc; ++i) {
          replaceVariables(newForSmnt,
                           *oldFunction.args[i],
                           *newFunction.args[i]);
        }

        newKernelSmnt.set(newForSmnt);
        root.scope.add(newFunction);

        return &newKernelSmnt;
      }

      void withLauncher::setupKernels() {
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
          replaceOccaFors(kernelSmnt);
          if (!success) return;
        }
      }

      void withLauncher::replaceOccaFors(functionDeclStatement &kernelSmnt) {
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


      void withLauncher::replaceOccaFor(forStatement &forSmnt) {
        oklForStatement oklForSmnt(forSmnt);

        std::string iteratorName;
        const int loopIndex = oklForSmnt.oklLoopIndex();
        if (oklForSmnt.isOuterLoop()) {
          iteratorName = getOuterIterator(loopIndex);
        } else {
          iteratorName = getInnerIterator(loopIndex);
        }

        identifierToken iteratorSource(oklForSmnt.iterator->source->origin,
                                       iteratorName);
        identifierNode iterator(&iteratorSource,
                                iteratorName);

        // Create iterator declaration
        variableDeclaration decl;
        decl.variable = oklForSmnt.iterator;
        decl.value = oklForSmnt.makeDeclarationValue(iterator);

        // Replace for-loops with blocks
        const int childIndex = forSmnt.childIndex();
        blockStatement &blockSmnt = *(new blockStatement(forSmnt.up,
                                                         forSmnt.source));
        blockSmnt.swap(forSmnt);
        blockSmnt.up->children[childIndex] = &blockSmnt;

        // Add declaration before block
        declarationStatement &declSmnt = (
          *(new declarationStatement(blockSmnt.up))
        );
        declSmnt.declarations.push_back(decl);

        blockSmnt.addFirst(declSmnt);
        delete &forSmnt;
      }
    }
  }
}
