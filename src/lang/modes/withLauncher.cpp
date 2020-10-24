#include <occa/tools/string.hpp>
#include <occa/lang/modes/withLauncher.hpp>
#include <occa/lang/modes/okl.hpp>
#include <occa/lang/modes/oklForStatement.hpp>
#include <occa/lang/builtins/attributes.hpp>
#include <occa/lang/builtins/types.hpp>
#include <occa/lang/expr.hpp>

namespace occa {
  namespace lang {
    namespace okl {
      withLauncher::withLauncher(const occa::properties &settings_) :
        parser_t(settings_),
        launcherParser(settings["launcher"]),
        memoryType(NULL) {
        launcherParser.settings["okl/validate"] = false;
      }

      //---[ Public ]-------------------
      bool withLauncher::succeeded() const {
        return (success && launcherParser.success);
      }

      void withLauncher::writeLauncherSourceToFile(const std::string &filename) const {
        launcherParser.writeToFile(filename);
      }
      //================================

      void withLauncher::launcherClear() {
        launcherParser.onClear();

        // Will get deleted by the parser
        memoryType = NULL;
      }

      void withLauncher::afterParsing() {
        if (!success) return;
        if (settings.get("okl/validate", true)) {
          success = kernelsAreValid(root);
        }

        if (!memoryType) {
          identifierToken memoryTypeSource(originSource::builtin,
                                           "occa::modeMemory_t");
          memoryType = new typedef_t(vartype_t(),
                                     memoryTypeSource);
        }

        root.addToScope(*memoryType);

        if (!success) return;
        setOklLoopIndices();

        if (!success) return;
        setupLauncherParser();

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

      void withLauncher::setOklLoopIndices() {
        root.children
            .forEachKernelStatement(okl::setOklLoopIndices);
      }

      void withLauncher::setupLauncherParser() {
        // Clone source
        blockStatement &rootClone = (blockStatement&) root.clone();

        launcherParser.root.swap(rootClone);
        delete &rootClone;
        launcherParser.setupKernels();

        // Remove outer loops
        launcherParser.root.children
            .forEachKernelStatement([&](functionDeclStatement &kernelSmnt) {
                removeLauncherOuterLoops(kernelSmnt);
                setupLauncherKernelArgs(kernelSmnt);
              });

        setupLauncherHeaders();
      }

      void withLauncher::removeLauncherOuterLoops(functionDeclStatement &kernelSmnt) {
        int kernelIndex = 0;

        kernelSmnt.children
            .flatFilterByAttribute("outer")
            .filterByStatementType(statementType::for_)
            .forEach([&](statement_t *smnt) {
                forStatement &forSmnt = (forStatement&) *smnt;
                if (isOuterMostOuterLoop(forSmnt)) {
                  setKernelLaunch(kernelSmnt,
                                  forSmnt,
                                  kernelIndex++);
                }
              });
      }

      bool withLauncher::isOuterMostOuterLoop(forStatement &forSmnt) {
        return isOuterMostOklLoop(forSmnt, "outer");
      }

      bool withLauncher::isOuterMostInnerLoop(forStatement &forSmnt) {
        return isOuterMostOklLoop(forSmnt, "inner");
      }

      bool withLauncher::isOuterMostOklLoop(forStatement &forSmnt,
                                            const std::string &attr) {
        statement_t *smnt = forSmnt.up;
        while (smnt) {
          if ((smnt->type() & statementType::for_)
              && smnt->hasAttribute(attr)) {
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

        statementArray path = oklForStatement::getOklLoopPath(*innerSmnt);

        // Create block in case there are duplicate variable names
        blockStatement &launchBlock = (
          *new blockStatement(forSmnt.up, forSmnt.source)
        );
        forSmnt.up->addBefore(forSmnt, launchBlock);

        // Get max count
        int outerCount = 0;
        int innerCount = 0;
        const int pathCount = (int) path.length();
        for (int i = 0; i < pathCount; ++i) {
          forStatement &pathSmnt = *((forStatement*) path[i]);
          oklForStatement oklForSmnt(pathSmnt);
          if (!oklForSmnt.isValid()) {
            success = false;
            return;
          }
          const bool isOuter = pathSmnt.hasAttribute("outer");
          outerCount += isOuter;
          innerCount += !isOuter;
        }
        const int outerDims = outerCount;
        const int innerDims = innerCount;

        // TODO 1.1: Properly fix this
        for (int i = 0; i < pathCount; ++i) {
          forStatement &pathSmnt = *((forStatement*) path[i]);
          oklForStatement oklForSmnt(pathSmnt);

          launchBlock.add(pathSmnt.init->clone(&launchBlock));

          const bool isOuter = pathSmnt.hasAttribute("outer");
          outerCount -= isOuter;
          innerCount -= !isOuter;

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
        }

        // Kernel launch
        std::string kernelLaunch = "kernel(";
        function_t &func = kernelSmnt.function();
        const int argCount = (int) func.args.size();
        for (int i = 0; i < argCount; ++i) {
          variable_t &arg = *(func.args[i]);
          if (i) {
            kernelLaunch += ", ";
          }
          kernelLaunch += arg.name();
        }
        kernelLaunch += ");";

        array<std::string> initSource = {
          "occa::dim outer, inner;",
          "outer.dims = " + occa::toString(outerDims) + ";",
          "inner.dims = " + occa::toString(innerDims) + ";"
        };

        array<std::string> kernelLaunchSource = {
          "occa::kernel kernel(deviceKernel[" + occa::toString(kernelIndex) + "]);",
          "kernel.setRunDims(outer, inner);",
          kernelLaunch
        };

        // We need to insert them at the top in reverse order
        initSource.reverse().forEach([&](std::string str) {
            launchBlock.addFirst(
              *new sourceCodeStatement(&launchBlock, forSmnt.source, str)
            );
          });

        kernelLaunchSource.forEach([&](std::string str) {
            launchBlock.add(
              *new sourceCodeStatement(&launchBlock, forSmnt.source, str)
            );
          });

        forSmnt.removeFromParent();

        // TODO 1.1: Delete after properly cloning the declaration statement
        // delete &forSmnt;
      }

      void withLauncher::setupLauncherKernelArgs(functionDeclStatement &kernelSmnt) {
        function_t &func = kernelSmnt.function();

        // Create new types
        identifierToken kernelTypeSource(kernelSmnt.source->origin,
                                         "occa::modeKernel_t");
        type_t &kernelType = *(new typedef_t(vartype_t(),
                                             kernelTypeSource));

        // Convert pointer arguments to modeMemory_t
        int argCount = (int) func.args.size();
        for (int i = 0; i < argCount; ++i) {
          variable_t &arg = *(func.args[i]);
          arg.vartype = arg.vartype.flatten();
          if (arg.vartype.isPointerType()) {
            arg.vartype = *memoryType;
            arg += pointer_t();
          }
        }

        // Add kernel array as the first argument
        identifierToken kernelVarSource(kernelSmnt.source->origin,
                                        "*deviceKernel");
        variable_t &kernelVar = *(new variable_t(kernelType,
                                                 &kernelVarSource));
        kernelVar += pointer_t();

        func.args.insert(func.args.begin(),
                         &kernelVar);

        kernelSmnt.addToScope(kernelType);
        kernelSmnt.addToScope(kernelVar);
      }

      void withLauncher::setupLauncherHeaders() {
        // TODO 1.1: Remove hack after methods are properly added
        const int headerCount = 2;
        std::string headers[headerCount] = {
          "include <occa/core/base.hpp>",
          "include <occa/modes/serial/kernel.hpp>"
        };
        for (int i = 0; i < headerCount; ++i) {
          std::string header = headers[i];
          directiveToken token(root.source->origin,
                               header);
          launcherParser.root.addFirst(
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
        int maxLevel = -1;
        forStatement *innerMostInnerLoop = NULL;

        forSmnt.children
            .flatFilterByAttribute("inner")
            .filterByStatementType(statementType::for_)
            .forEach([&](statement_t *smnt) {
                forStatement &innerSmnt = (forStatement&) *smnt;
                const int level = getInnerLoopLevel(innerSmnt);
                if (level > maxLevel) {
                  maxLevel = level;
                  innerMostInnerLoop = &innerSmnt;
                }
              });

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
        root.children
            .forEachKernelStatement([&](functionDeclStatement &kernelSmnt) {
                splitKernel(kernelSmnt);
              });
      }

      void withLauncher::splitKernel(functionDeclStatement &kernelSmnt) {
        statementArray newKernelSmnts;
        int kernelIndex = 0;

        kernelSmnt.children
            .flatFilterByAttribute("outer")
            .filterByStatementType(statementType::for_)
            .forEach([&](statement_t *smnt) {
                forStatement &forSmnt = (forStatement&) *smnt;

                if (isOuterMostOuterLoop(forSmnt)) {
                  newKernelSmnts.push(
                    extractLoopAsKernel(kernelSmnt,
                                        forSmnt,
                                        kernelIndex++)
                  );
                }
              });

        int smntIndex = kernelSmnt.childIndex();
        for (int i = (kernelIndex - 1); i >= 0; --i) {
          root.add(*(newKernelSmnts[i]),
                   smntIndex);
        }

        root.remove(kernelSmnt);
        root.removeFromScope(kernelSmnt.function().name(), true);

        // TODO 1.1: Find out what causes segfault here
        // delete &kernelSmnt;
      }

      statement_t* withLauncher::extractLoopAsKernel(functionDeclStatement &kernelSmnt,
                                                     forStatement &forSmnt,
                                                     const int kernelIndex) {

        function_t &oldFunction = kernelSmnt.function();
        function_t &newFunction = (function_t&) oldFunction.clone();
        std::stringstream ss;
        ss << +"_occa_" << newFunction.name() << "_" << kernelIndex;
        newFunction.source->value = ss.str();

        functionDeclStatement &newKernelSmnt = *(
          new functionDeclStatement(&root,
                                    newFunction)
        );
        newKernelSmnt.attributes = kernelSmnt.attributes;
        newKernelSmnt.addFunctionToParentScope();

        // Clone for-loop and replace argument variables
        forStatement &newForSmnt = (forStatement&) forSmnt.clone();
        newKernelSmnt.set(newForSmnt);

        const int argc = (int) newFunction.args.size();
        for (int i = 0; i < argc; ++i) {
          newForSmnt.replaceVariable(
            *oldFunction.args[i],
            *newFunction.args[i]
          );
        }

        return &newKernelSmnt;
      }

      void withLauncher::setupKernels() {
        root.children
            .forEachKernelStatement([&](functionDeclStatement &kernelSmnt) {
                setupOccaFors(kernelSmnt);
              });
      }

      void withLauncher::setupOccaFors(functionDeclStatement &kernelSmnt) {
        kernelSmnt.children
            .flatFilterByAttribute("outer")
            .filterByStatementType(statementType::for_)
            .forEach([&](statement_t *smnt) {
                forStatement &outerSmnt = (forStatement&) *smnt;
                replaceOccaFor(outerSmnt);
              });

        const bool applyBarriers = usesBarriers();

        kernelSmnt.children
            .flatFilterByAttribute("inner")
            .filterByStatementType(statementType::for_)
            .forEach([&](statement_t *smnt) {
                forStatement &innerSmnt = (forStatement&) *smnt;

                // TODO 1.1: Only apply barriers when needed in the last inner-loop
                if (applyBarriers &&
                    isOuterMostInnerLoop(innerSmnt)) {
                  addBarriersAfterInnerLoop(innerSmnt);
                }

                replaceOccaFor(innerSmnt);
              });
      }

      void withLauncher::addBarriersAfterInnerLoop(forStatement &forSmnt) {
        const bool noSharedWrites = (
          forSmnt.children
          .flatFilter([&](smntExprNode smntExpr) {
              return writesToShared(*smntExpr.node);
            })
          .isEmpty()
        );

        if (noSharedWrites) {
          return;
        }

        statement_t &barrierSmnt = (
          *(new emptyStatement(forSmnt.up,
                               forSmnt.source))
        );

        identifierToken barrierToken(forSmnt.source->origin,
                                     "barrier");

        barrierSmnt.attributes["barrier"] = (
          attributeToken_t(*(getAttribute("barrier")),
                           barrierToken)
        );

        forSmnt.up->addAfter(forSmnt,
                             barrierSmnt);
      }

      bool withLauncher::writesToShared(exprNode &expr) {
        // TODO 1.1: Propertly check read<-->write or write<-->write ordering
        // exprOpNode &opNode = (exprOpNode&) expr;
        // if (!(opNode.opType() & (operatorType::increment |
        //                          operatorType::decrement |
        //                          operatorType::assignment))) {
        //   return false;
        // }

        // Get updated variable
        variable_t *var = expr.getVariable();
        return (var &&
                var->hasAttribute("shared"));
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
        variableDeclaration decl(
          *oklForSmnt.iterator,
          oklForSmnt.makeDeclarationValue(iterator)
        );

        // Replace for-loops with blocks
        const int childIndex = forSmnt.childIndex();
        blockStatement &blockSmnt = *(new blockStatement(forSmnt.up,
                                                         forSmnt.source));
        blockSmnt.swap(forSmnt);
        blockSmnt.up->children[childIndex] = &blockSmnt;

        // Add declaration before block
        declarationStatement &declSmnt = (
          *(new declarationStatement(blockSmnt.up,
                                     forSmnt.source))
        );
        declSmnt.declarations.push_back(decl);

        blockSmnt.addFirst(declSmnt);
        delete &forSmnt;
      }

      bool withLauncher::usesBarriers() {
        return true;
      }
    }
  }
}
