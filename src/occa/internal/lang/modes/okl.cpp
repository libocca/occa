#include <map>

#include <occa/internal/lang/modes/okl.hpp>
#include <occa/internal/lang/variable.hpp>
#include <occa/internal/lang/parser.hpp>
#include <occa/internal/lang/builtins/attributes.hpp>
#include <occa/internal/lang/builtins/types.hpp>
#include <occa/internal/lang/expr.hpp>
#include <occa/internal/lang/modes/oklForStatement.hpp>
#include <occa/internal/lang/statement/blockStatement.hpp>

namespace occa {
  namespace lang {
    namespace okl {
      bool kernelsAreValid(blockStatement &root) {
        statementArray kernelSmnts = (
          root.children.getKernelStatements()
        );

        if (!kernelSmnts.length()) {
          occa::printError("No [@kernel] functions found");
          return false;
        }

        // Make sure no kernels fail the checks needed
        return kernelSmnts
            .filter([&](statement_t *kernelSmnt) {
                return !kernelIsValid((functionDeclStatement&) *kernelSmnt);
              })
            .isEmpty();
      }

      bool kernelIsValid(functionDeclStatement &kernelSmnt) {
        return (
          kernelHasValidReturnType(kernelSmnt)
          && kernelHasValidOklLoops(kernelSmnt)
          && kernelHasValidSharedAndExclusiveDeclarations(kernelSmnt)
          && kernelHasValidLoopBreakAndContinue(kernelSmnt)
        );
      }

      bool kernelHasValidReturnType(functionDeclStatement &kernelSmnt) {
        vartype_t &returnType = kernelSmnt.function().returnType;

        if (returnType.qualifiers.size() || (*returnType.type != void_)) {
          returnType.printError(
            "[@kernel] functions must have a [void] return type"
          );
          return false;
        }

        return true;
      }

      bool kernelHasValidOklLoops(functionDeclStatement &kernelSmnt) {
        array<statementArray> loopPaths;
        statementArray outerLoops, innerLoops;

        forOklForLoopStatements(
          kernelSmnt,
          [&](forStatement &forSmnt, const std::string attr, const statementArray &path) {
            if (attr == "outer") {
              outerLoops.push(&forSmnt);
            } else {
              innerLoops.push(&forSmnt);
            }

            // Filter to only have @outer and @inner loops
            // Push the current statement in the loop path
            statementArray loopPath = (
              path.filter([&](statement_t *smnt) {
                  return isOklForLoop(smnt);
                })
            );
            loopPath.push(&forSmnt);

            loopPaths.push(loopPath);
          });

        // Make sure we have @outer and @inner for-loops
        if (!outerLoops.length()) {
          kernelSmnt.printError("[@kernel] requires at least one [@outer] for-loop");
          return false;
        }
        if (!innerLoops.length()) {
          kernelSmnt.printError("[@kernel] requires at least one [@inner] for-loop");
          return false;
        }

        // Make sure the forStatements are valid
        for (statement_t *smnt : outerLoops) {
          if (!oklForStatement::isValid(*((forStatement*) smnt), "outer")) {
            return false;
          }
        }
        for (statement_t *smnt : innerLoops) {
          if (!oklForStatement::isValid(*((forStatement*) smnt), "inner")) {
            return false;
          }
        }

        // When the ordered loop paths are reversed, we'll find the inner-most inner/outer loop
        // before the parent paths.
        //
        // To check if a loop is the inner-most okl-for-loop is to check if
        // - Loop path starts with the previous okl loop path
        //   loopPaths[i]     = [@outer_1, @outer_2, @inner_1]
        //   loopPaths[i + 1] = [@outer_1, @outer_2, @inner_1, @inner_2] (starts with loopPaths[i])
        //
        // - Next loop path doesn't start with the current loop path
        //   loopPaths[i]     = [@outer_1, @outer_2, @inner_1]
        //   loopPaths[i + 1] = [@outer_1, @outer_2, @inner_2] (doesn't start with loopPaths[i])
        //
        // We reverse the paths so it's easy to check the "next" loop path by saving the
        // previous path in the reversed array
        statementArray nextLoopPath;

        array<statementArray> innerMostPaths = (
          loopPaths
          .reverse()
          .filter([&](statementArray path) {
              const bool isInnerMost = !nextLoopPath.startsWith(path);

              nextLoopPath = path;

              return isInnerMost;
            })
        );

        statement_t *currentOuterMostOuterLoop = NULL;
        int currentInnerLoopCount = 0;
        int currentOuterLoopCount = 0;

        for (auto &path : innerMostPaths) {
          statement_t *outerMostOuterLoop = path[0];

          int innerLoopCount, outerLoopCount;
          if (!pathHasValidOklLoopOrdering(path, innerLoopCount, outerLoopCount)) {
            return false;
          }

          if (outerMostOuterLoop != currentOuterMostOuterLoop) {
            // We ran into a different outer-most @outer loop
            // Reset the expected loop counts
            currentOuterMostOuterLoop = outerMostOuterLoop;
            currentInnerLoopCount = innerLoopCount;
            currentOuterLoopCount = outerLoopCount;
          } else {
            // Make sure the @outer and @inner counts are still the same
            if (currentInnerLoopCount != innerLoopCount) {
              path.last()->printError("Mismatch of [@inner] loops");
              return false;
            }
            if (currentOuterLoopCount != outerLoopCount) {
              path.last()->printError("Mismatch of [@outer] loops");
              return false;
            }
          }
        }

        return true;
      }

      bool pathHasValidOklLoopOrdering(statementArray &loopPath,
                                       int &innerLoopCount,
                                       int &outerLoopCount) {
        // Keep count
        innerLoopCount = 0;
        outerLoopCount = 0;

        for (auto smnt : loopPath) {
          if (smnt->hasAttribute("outer")) {
            ++outerLoopCount;
            if (innerLoopCount) {
              smnt->printError("Cannot have [@outer] loop inside an [@inner] loop");
              return false;
            }
          } else if (smnt->hasAttribute("inner")) {
            ++innerLoopCount;
            if (!outerLoopCount) {
              smnt->printError("Cannot have [@inner] loop outside of an [@outer] loop");
              return false;
            }
          }
        }

        // The !outerLoopCount is covered inside the "inner" check above
        if (!innerLoopCount) {
          for (auto smnt : loopPath) {
            if (smnt->hasAttribute("outer")) {
              smnt->printError("Missing an [@inner] loop");
              return false;
            }
          }
        }

        return true;
      }

      bool kernelHasValidSharedAndExclusiveDeclarations(functionDeclStatement &kernelSmnt) {
        bool isValid = true;

        statementArray::from(kernelSmnt)
            .flatFilterByStatementType(
              statementType::declaration
              | statementType::expression
            )
            .flatFilterByExprType(exprNodeType::variable)
            .forEach([&](smntExprNode smntExpr) {
                statement_t *smnt = smntExpr.smnt;
                exprNode *node = smntExpr.node;

                variable_t &var = ((variableNode*) node)->value;

                const bool isShared = var.hasAttribute("shared");
                const bool isExclusive = var.hasAttribute("exclusive");

                if (!isShared && !isExclusive) {
                  return;
                }

                const bool varIsBeingDeclared = (
                  (smnt->type() & statementType::declaration)
                  && ((declarationStatement*) smnt)->declaresVariable(var)
                );

                if (varIsBeingDeclared && isShared) {
                  isValid &= hasProperSharedArrayDeclaration(var);
                }

                // Definition must be outside of an inner loop
                isValid &= hasProperSharedOrExclusiveUsage(
                  smnt,
                  isShared ? "shared" : "exclusive",
                  varIsBeingDeclared
                );
              });

        return isValid;
      }

      bool hasProperSharedArrayDeclaration(variable_t &var) {
        vartype_t &vartype = var.vartype;

        if (!vartype.arrays.size()) {
          var.printError("[@shared] variables must be arrays");
          return false;
        }

        for (auto arr : vartype.arrays) {
          if (!arr.size ||
              !arr.size->canEvaluate()) {
            arr.printError("[@shared] variables must have sizes known at compile-time");
            return false;
          }
        }

        return true;
      }

      bool hasProperSharedOrExclusiveUsage(
        statement_t *smnt,
        const std::string &attrName,
        bool varIsBeingDeclared
      ) {
        bool inOuter = false;
        bool inInner = false;

        statement_t *pathSmnt = smnt;
        while (pathSmnt) {
          if (pathSmnt->type() & statementType::for_) {
            inInner |= pathSmnt->hasAttribute("inner");
            inOuter |= pathSmnt->hasAttribute("outer");
          }
          pathSmnt = pathSmnt->up;
        }

        // TODO: Make sure it's in the inner-most @inner loop
        if (varIsBeingDeclared) {
          if (inInner) {
            smnt->printError("Cannot define [@" + attrName + "] variables inside"
                             " an [@inner] loop");
            return false;
          }
          if (!inOuter) {
            smnt->printError("Must define [@" + attrName + "] variables between"
                             " [@outer] and [@inner] loops");
            return false;
          }
        } else if (!inInner) {
          smnt->printError("Cannot use [@" + attrName + "] variables outside"
                           " an [@inner] loop");
          return false;
        }

        return true;
      }

      bool kernelHasValidLoopBreakAndContinue(functionDeclStatement &kernelSmnt) {
        // No break or continue directly inside @outer/@inner loops
        // It's ok inside regular loops inside @outer/@inner
        return (
          statementArray::from(kernelSmnt)
          .flatFilterByStatementType(
            statementType::continue_
            | statementType::break_
          )
          .filter([&](statement_t *smnt) {
              statement_t *parentSmnt = smnt->up;

              while (parentSmnt) {
                const int sType = parentSmnt->type();

                // Break/continue is for a non-okl while/switch statement
                if (sType & (statementType::while_ |
                             statementType::switch_)) {
                  return false;
                }

                if (!(sType & statementType::for_)) {
                  parentSmnt = parentSmnt->up;
                  continue;
                }

                if (parentSmnt->hasAttribute("inner")) {
                  smnt->printError("Statement cannot be directly inside an [@inner] loop");
                  parentSmnt->printError("[@inner] loop is here");
                  return true;
                }
                if (parentSmnt->hasAttribute("outer")) {
                  smnt->printError("Statement cannot be directly inside an [@outer] loop");
                  parentSmnt->printError("[@outer] loop is here");
                  return true;
                }

                break;
              }

              // Break/continue is for a non-okl for-loop statement
              return false;
            })
          .isEmpty()
        );
      }

      //---[ Helper Methods ]-----------
      bool isOklForLoop(statement_t *smnt) {
        std::string oklAttr;
        return isOklForLoop(smnt, oklAttr);
      }

      bool isOklForLoop(statement_t *smnt, std::string &oklAttr) {
        // Only checking for for-loops
        if (!(smnt->type() & statementType::for_)) {
          return false;
        };

        if (smnt->hasAttribute("outer")) {
          oklAttr = "outer";
          return true;
        }

        if (smnt->hasAttribute("inner")) {
          oklAttr = "inner";
          return true;
        }

        return false;
      }

      void forOklForLoopStatements(statement_t &root, oklForVoidCallback func) {
        statementArray::from(root)
            .nestedForEach([&](statement_t *smnt, const statementArray &path) {
                std::string oklAttr;

                if (!isOklForLoop(smnt, oklAttr)) {
                  return;
                }

                func((forStatement&) *smnt, oklAttr, path);
              });
      }
      //================================

      //---[ Transformations ]----------
      void addOklAttributes(parser_t &parser) {
        parser.addAttribute<attributes::barrier>();
        parser.addAttribute<attributes::exclusive>();
        parser.addAttribute<attributes::inner>();
        parser.addAttribute<attributes::kernel>();
        parser.addAttribute<attributes::outer>();
        parser.addAttribute<attributes::shared>();
        parser.addAttribute<attributes::maxInnerDims>();
        parser.addAttribute<attributes::noBarrier>();
      }

      void setOklLoopIndices(functionDeclStatement &kernelSmnt) {
        auto func = [&](forStatement &forSmnt,
                        const std::string attr,
                        const statementArray &path) {
          attributeToken_t &oklAttr = forSmnt.attributes[attr];
          if (oklAttr.args.size()) {
            return;
          }

          const int loopIndex = oklForStatement::getOklLoopIndex(forSmnt, attr);

          oklAttr.args.push_back(
            new primitiveNode(oklAttr.source,
                              loopIndex)
          );
        };

        forOklForLoopStatements(kernelSmnt, func);
      }
      //================================
    }
  }
}
