#include <occa/internal/utils/string.hpp>
#include <occa/internal/lang/modes/withLauncher.hpp>
#include <occa/internal/lang/modes/okl.hpp>
#include <occa/internal/lang/modes/oklForStatement.hpp>
#include <occa/internal/lang/builtins/attributes.hpp>
#include <occa/internal/lang/builtins/types.hpp>
#include <occa/internal/lang/expr.hpp>

namespace occa {
  namespace lang {
    namespace okl {
      static const std::string modeMemoryTypeName = "occa::modeMemory_t";
      static const std::string modeKernelTypeName = "occa::modeKernel_t";

      withLauncher::withLauncher(const occa::json &settings_) :
        parser_t(settings_),
        launcherParser(settings["launcher"]) {
        launcherParser.settings["okl/validate"] = false;
        add_barriers = settings.get("okl/add_barriers", true);
      }

      //---[ Public ]-------------------
      bool withLauncher::succeeded() const {
        return (success && launcherParser.success);
      }

      void withLauncher::writeLauncherSourceToFile(const std::string &filename) const {
        io::stageFile(
          filename,
          true,
          [&](const std::string &tempFilename) -> bool {
            launcherParser.writeToFile(tempFilename);
            return true;
          }
        );
      }
      //================================

      void withLauncher::launcherClear() {
        launcherParser.clear();
      }

      void withLauncher::afterParsing() {
        if (!success) return;
        if (settings.get("okl/validate", true)) {
          success = kernelsAreValid(root);
        }

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

      type_t& withLauncher::getMemoryModeType() {
        return *launcherParser.root.getScopeType(modeMemoryTypeName);
      }

      type_t& withLauncher::getKernelModeType() {
        return *launcherParser.root.getScopeType(modeKernelTypeName);
      }

      void withLauncher::setOklLoopIndices() {
        root.children
          .forEachKernelStatement(okl::setOklLoopIndices);
      }

      void withLauncher::setupLauncherParser() {
        // Clone source
        blockStatement &rootClone = (blockStatement&) root.clone();

        launcherParser.root.swap(rootClone);
        delete &rootClone;

        // Add occa::mode* types
        identifierToken memoryTypeSource(originSource::builtin,
                                         modeMemoryTypeName);

        launcherParser.root.addToScope(
          *(new typedef_t(vartype_t(),
                          memoryTypeSource))
        );

        identifierToken kernelTypeSource(originSource::builtin,
                                         modeKernelTypeName);

        launcherParser.root.addToScope(
          *(new typedef_t(vartype_t(),
                          kernelTypeSource))
        );

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

        statementArray::from(kernelSmnt)
          .flatFilterByAttribute("outer")
          .filterByStatementType(statementType::for_)
          .filter([&](statement_t *smnt) {
            return isOuterMostOuterLoop((forStatement&) *smnt);
          })
          .forEach([&](statement_t *smnt) {
            setKernelLaunch(kernelSmnt,
                            (forStatement&) *smnt,
                            kernelIndex++);
          });

        kernelSmnt.updateIdentifierReferences();
        kernelSmnt.updateVariableReferences();
      }

      bool withLauncher::isOuterMostOuterLoop(forStatement &forSmnt) {
        return isOuterMostOklLoop(forSmnt, "outer");
      }

      bool withLauncher::isOuterMostInnerLoop(forStatement &forSmnt) {
        return isOuterMostOklLoop(forSmnt, "inner");
      }

      bool withLauncher::isOuterMostOklLoop(forStatement &forSmnt,
                                            const std::string &attr) {
        for (auto &parentSmnt : forSmnt.getParentPath()) {
          if (parentSmnt->type() & statementType::for_
              && parentSmnt->hasAttribute(attr)) {
            return false;
          }
        }
        return true;
      }

      bool withLauncher::isLastInnerLoop(forStatement &forSmnt) {
        blockStatement &parent = *(forSmnt.up);
        for(int smntIndex = forSmnt.childIndex()+1; smntIndex<parent.size(); smntIndex++) {
          if (statementArray::from(*parent[smntIndex])
                .flatFilterByAttribute("inner")
                .length()) {
            return false;
          }
        }
        return true;
      }

      bool withLauncher::isInsideLoop(forStatement &forSmnt) {
        for (auto &parentSmnt : forSmnt.getParentPath()) {
          if (parentSmnt->type() & (statementType::for_ | statementType::while_)) {
            return true;
          }
        }
        return false;
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
        forSmnt.replaceWith(launchBlock);

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
          "occa::kernel kernel(deviceKernels[" + occa::toString(kernelIndex) + "]);",
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

        launchBlock.updateIdentifierReferences();
        launchBlock.updateVariableReferences();

        // TODO: Figure out which variables are being deleted
        // delete &forSmnt;
      }

      void withLauncher::setupLauncherKernelArgs(functionDeclStatement &kernelSmnt) {
        function_t &func = kernelSmnt.function();

        type_t &memoryType = getMemoryModeType();

        // Convert pointer arguments to modeMemory_t
        int argCount = (int) func.args.size();
        for (int i = 0; i < argCount; ++i) {
          variable_t &arg = *(func.args[i]);
          arg.vartype = arg.vartype.flatten();
          if (arg.vartype.isPointerType()) {
            arg.vartype = memoryType;
            arg += pointer_t();
          }
        }

        // Add kernel array as the first argument
        identifierToken kernelVarSource(kernelSmnt.source->origin,
                                        "deviceKernels");
        variable_t &kernelVar = *(new variable_t(getKernelModeType(),
                                                 &kernelVarSource));
        kernelVar += pointer_t();
        kernelVar += pointer_t();

        func.args.insert(func.args.begin(),
                         &kernelVar);

        kernelSmnt.addToScope(kernelVar);

        kernelSmnt.updateVariableReferences();
      }

      void withLauncher::setupLauncherHeaders() {
        directiveToken token(root.source->origin,
                             "include <occa/core/kernel.hpp>");
        launcherParser.root.addFirst(
          *(new directiveStatement(&root, token))
        );
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

        statementArray::from(forSmnt)
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

        statementArray::from(kernelSmnt)
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

        root.updateVariableReferences();
        launcherParser.root.updateVariableReferences();

        // TODO: Figure out which variables are being deleted
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

        // Clone for-loop and replace argument variables
        forStatement &newForSmnt = (forStatement&) forSmnt.clone();
        newKernelSmnt.set(newForSmnt);

        bool addLaunchBoundsAttribute{true};
        int kernelInnerDims[3] = {1,1,1};
        if (newForSmnt.hasAttribute("max_inner_dims")) {
          attributeToken_t& attr = newForSmnt.attributes["max_inner_dims"];      

          for(size_t i=0; i < attr.args.size(); ++i) {
            exprNode* expr = attr.args[i].expr;
            primitive value = expr->evaluate();
            kernelInnerDims[i] = value; 
          } 
        } else {
          //Programmer hasn't specified launch bounds.
          //If they are known at compile time, set them.
          forStatement *innerSmnt = getInnerMostInnerLoop(newForSmnt);
          statementArray path = oklForStatement::getOklLoopPath(*innerSmnt);

          int innerIndex;
          const int pathCount = (int) path.length();
          for (int i = 0; i < pathCount; ++i) {
            forStatement &pathSmnt = *((forStatement*) path[i]);
            oklForStatement oklForSmnt(pathSmnt);

            if(pathSmnt.hasAttribute("inner")) {
              innerIndex = oklForSmnt.oklLoopIndex();
              if(oklForSmnt.getIterationCount()->canEvaluate()) {
                kernelInnerDims[innerIndex] = (int) oklForSmnt.getIterationCount()->evaluate();
              } else { 
                std::string s = oklForSmnt.getIterationCount()->toString();
                if(s.find("_occa_tiled_") != std::string::npos) {
                  size_t tile_size = s.find_first_of("123456789");
                  OCCA_ERROR("@tile size is undefined!",tile_size != std::string::npos);
                  kernelInnerDims[innerIndex] = std::stoi(s.substr(tile_size));
                } else {
                  //loop bounds are unknown at compile time
                  addLaunchBoundsAttribute=false;
                  break;
                }
              }
            }
          }
        }

        if(addLaunchBoundsAttribute) {
          std::string lbAttr = launchBoundsAttribute(kernelInnerDims);
          qualifier_t& boundQualifier = *(new qualifier_t(lbAttr,qualifierType::custom));
          function_t& function = newKernelSmnt.function();
          function.returnType.add(1, boundQualifier);
        }

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
        statementArray::from(kernelSmnt)
          .flatFilterByAttribute("outer")
          .filterByStatementType(statementType::for_)
          .forEach([&](statement_t *smnt) {
            forStatement &outerSmnt = (forStatement&) *smnt;
            replaceOccaFor(outerSmnt);
          });

        if (usesBarriers()) {
          statementArray::from(kernelSmnt)
            .flatFilterByAttribute("inner")
            .filterByStatementType(statementType::for_)
            .forEach([&](statement_t *smnt) {
              forStatement &innerSmnt = (forStatement&) *smnt;

              //Only apply barriers when needed in the last inner-loop
              if (isOuterMostInnerLoop(innerSmnt)
                  && (!isLastInnerLoop(innerSmnt) || isInsideLoop(innerSmnt))
                  && !(innerSmnt.hasAttribute("nobarrier"))
                 ) addBarriersAfterInnerLoop(innerSmnt);
            });
        }

        statementArray::from(kernelSmnt)
          .flatFilterByAttribute("inner")
          .filterByStatementType(statementType::for_)
          .forEach([&](statement_t *smnt) {
            forStatement &innerSmnt = (forStatement&) *smnt;
            replaceOccaFor(innerSmnt);
          });
      }

      void withLauncher::addBarriersAfterInnerLoop(forStatement &forSmnt) {
        const bool noSharedWrites = (
          statementArray::from(forSmnt)
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
        return add_barriers;
      }
    }
  }
}
