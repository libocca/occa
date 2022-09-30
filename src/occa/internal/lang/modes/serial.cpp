#include <set>

#include <occa/internal/lang/modes/serial.hpp>
#include <occa/internal/lang/modes/okl.hpp>
#include <occa/internal/lang/modes/oklForStatement.hpp>
#include <occa/internal/lang/builtins/types.hpp>
#include <occa/internal/lang/expr.hpp>

namespace occa {
  namespace lang {
    namespace okl {
      const std::string serialParser::exclusiveIndexName = "_occa_exclusive_index";

      serialParser::serialParser(const occa::json &settings_) :
        parser_t(settings_) {

        okl::addOklAttributes(*this);
      }

      void serialParser::onClear() {}

      void serialParser::afterParsing() {
        if (!success) return;
        if (settings.get("okl/validate", true)) {
          success = kernelsAreValid(root);
        }

        if (!success) return;
        setupKernels();

        if (!success) return;
        setupExclusives();
      }

      void serialParser::setupHeaders() {
        strVector headers;
        const bool includingStd = settings.get("serial/include_std", true);
        headers.push_back("include <occa.hpp>\n");
        if (includingStd) {
          headers.push_back("include <stdint.h>");
          headers.push_back("include <cstdlib>");
          headers.push_back("include <cstdio>");
          headers.push_back("include <cmath>");
        }

        const int headerCount = (int) headers.size();
        for (int i = 0; i < headerCount; ++i) {
          std::string header = headers[i];
          // TODO 1.1: Remove hack after methods are properly added
          if (i == 0) {
            if (includingStd) {
              header += "\nusing namespace std;";
            }
            header += "\nusing namespace occa;";
          }
          directiveToken token(root.source->origin,
                               header);
          root.addFirst(
            *(new directiveStatement(&root, token))
          );
        }
      }

      void serialParser::setupKernels() {
        setupHeaders();
        if (!success) return;

        root.children.forEachKernelStatement(setupKernel);
      }

      void serialParser::setupKernel(functionDeclStatement &kernelSmnt) {
        // @kernel -> extern "C"
        function_t &func = kernelSmnt.function();
        attributeToken_t &kernelAttr = kernelSmnt.attributes["kernel"];
        qualifiers_t &qualifiers = func.returnType.qualifiers;

#if OCCA_OS == OCCA_WINDOWS_OS
        // Add extern "C" [__declspec(dllexport)]
        qualifiers.addFirst(kernelAttr.source->origin,
                            dllexport_);
#endif
        qualifiers.addFirst(kernelAttr.source->origin,
                            externC);

        // Remove other externs
        if (qualifiers.has(extern_)) {
          qualifiers -= extern_;
        }
        if (qualifiers.has(externCpp)) {
          qualifiers -= externCpp;
        }
        // Pass non-pointer arguments by reference
        const int argCount = (int) func.args.size();
        for (int i = 0; i < argCount; ++i) {
          variable_t &arg = *(func.args[i]);
          vartype_t &type = arg.vartype;
          if ((type.isPointerType() ||
               type.referenceToken)) {
            continue;
          }
          operatorToken opToken(arg.source->origin,
                                op::bitAnd);
          type.setReferenceToken(&opToken);
        }
      }

      void serialParser::setupExclusives() {
        // Get @exclusive declarations
        bool hasExclusiveVariables = false;
        statementArray::from(root)
          .nestedForEachDeclaration([&](variableDeclaration &decl, declarationStatement &declSmnt) {
            if (decl.variable().hasAttribute("exclusive")) {
              hasExclusiveVariables = true;
              setupExclusiveDeclaration(declSmnt);
            }
          });
        if (!success) return;

        if (!hasExclusiveVariables) {
          return;
        }

        setupExclusiveIndices();
        if (!success) return;

        statementArray::from(root)
          .flatFilterByExprType(exprNodeType::variable, "exclusive")
          .inplaceMap([&](smntExprNode smntExpr) -> exprNode* {
            statement_t *smnt = smntExpr.smnt;
            variableNode &varNode = (variableNode&) *smntExpr.node;
            variable_t &var = varNode.value;

            if (
              (smnt->type() & statementType::declaration)
              && ((declarationStatement*) smnt)->declaresVariable(var)
            ) {
              defineExclusiveVariableAsArray((declarationStatement&) *smnt, var);
              return &varNode;
            }

            return addExclusiveVariableArrayAccessor(*smnt, varNode, var);
          });
      }

      void serialParser::setupExclusiveDeclaration(declarationStatement &declSmnt) {
        // Find inner-most outer loop
        statement_t *smnt = declSmnt.up;
        forStatement *innerMostOuterLoop = NULL;
        while (smnt) {
          if (smnt->hasAttribute("outer")) {
            innerMostOuterLoop = (forStatement*) smnt;
            break;
          }
          smnt = smnt->up;
        }

        // Check if index variable exists and is valid
        if (innerMostOuterLoop->hasDirectlyInScope(exclusiveIndexName)) {
          keyword_t &keyword = innerMostOuterLoop->getScopeKeyword(exclusiveIndexName);
          if (keyword.type() != keywordType::variable) {
            keyword.printError(exclusiveIndexName + " is a restricted OCCA keyword");
            success = false;
          }
          return;
        }

        // Create index variable and its declaration statement
        const fileOrigin &origin = innerMostOuterLoop->source->origin;
        identifierToken varSource(origin, exclusiveIndexName);
        variable_t *indexVar = new variable_t(
          vartype_t(identifierToken(origin, "int"), int_),
          &varSource
        );

        // Create declaration statement for index variable
        declarationStatement &indexDeclSmnt = *(new declarationStatement(innerMostOuterLoop,
                                                                         &varSource));
        innerMostOuterLoop->addFirst(indexDeclSmnt);
        // Add variable to decl + scope
        indexDeclSmnt.addDeclaration(*indexVar);
      }

      void serialParser::setupExclusiveIndices() {
        // Defines the exclusive index:
        //   int _occa_exclusive_index = 0;
        std::set<statement_t*> outerMostInnerLoops;

        // Increments the exlusive index:
        //   ++_occa_exclusive_index;
        std::set<statement_t*> innerMostInnerLoops;

        std::set<statement_t*> loopsWithExclusiveUsage;
        statementArray::from(root)
          .flatFilterByStatementType(statementType::for_, "inner")
          .forEach([&](statement_t *smnt) {
            const bool hasExclusiveUsage = smnt->hasInScope(exclusiveIndexName);
            if (!hasExclusiveUsage) {
              return;
            }

            loopsWithExclusiveUsage.insert(smnt);

            statementArray path = smnt->getParentPath();

            // Get outer-most inner loop
            bool isInnerMostInnerLoop = true;
            for (auto pathSmnt : path) {
              if (pathSmnt->hasAttribute("inner")) {
                outerMostInnerLoops.insert(pathSmnt);
                isInnerMostInnerLoop = false;
                break;
              }
            }

            if (isInnerMostInnerLoop) {
              outerMostInnerLoops.insert(smnt);
            }

            // Remove parent "inner" loops from the innerMostInnerLoops set
            innerMostInnerLoops.insert(smnt);
            for (auto pathSmnt : path) {
              if (pathSmnt->hasAttribute("inner")) {
                innerMostInnerLoops.erase(pathSmnt);
              }
            }
          });

        // Initialize the exclusive index to 0 before the outer-most inner loop
        for (auto smnt : outerMostInnerLoops) {
          forStatement &forSmnt = (forStatement&) *smnt;
          keyword_t &keyword = forSmnt.getScopeKeyword(exclusiveIndexName);
          variable_t &indexVar = ((variableKeyword&) keyword).variable;

          variableNode indexVarNode(forSmnt.source,
                                    indexVar);
          primitiveNode zeroNode(forSmnt.source,
                                 0);
          binaryOpNode assign(forSmnt.source,
                              op::assign,
                              indexVarNode,
                              zeroNode);

          forSmnt.up->addBefore(
            forSmnt,
            *(new expressionStatement(&forSmnt,
                                      *(assign.clone())))
          );
        }

        // Increment the exclusive index in the inner-most inner loop
        for (auto smnt : innerMostInnerLoops) {
          forStatement &forSmnt = (forStatement&) *smnt;
          keyword_t &keyword = forSmnt.getScopeKeyword(exclusiveIndexName);
          variable_t &indexVar = ((variableKeyword&) keyword).variable;

          variableNode indexVarNode(forSmnt.source,
                                    indexVar);
          leftUnaryOpNode increment(forSmnt.source,
                                    op::leftIncrement,
                                    indexVarNode);
          forSmnt.addLast(
            *(new expressionStatement(&forSmnt,
                                      *(increment.clone())))
          );
        }
      }

      int serialParser::getInnerLoopLevel(forStatement &forSmnt) {
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

      forStatement* serialParser::getInnerMostInnerLoop(forStatement &forSmnt) {
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

      void serialParser::defineExclusiveVariableAsArray(declarationStatement &declSmnt,
                                                        variable_t &var) {
        // Find outer-most outer loop
        statement_t *smnt = declSmnt.up;
        forStatement *outerMostOuterLoop = NULL;
        while (smnt) {
          if (smnt->hasAttribute("outer")) {
            outerMostOuterLoop = (forStatement*) smnt;
          }
          smnt = smnt->up;
        }

        // Check if outer loop has max_inner_dims set
        bool maxInnerDimsKnown{false};
        int maxInnerDims[3] = {1,1,1};
        if (outerMostOuterLoop->hasAttribute("max_inner_dims")) {
          maxInnerDimsKnown = true;
          attributeToken_t& attr = outerMostOuterLoop->attributes["max_inner_dims"];

          for(size_t i=0; i < attr.args.size(); ++i) {
            exprNode* expr = attr.args[i].expr;
            primitive value = expr->evaluate();
            maxInnerDims[i] = value;
          }
        }

        //Check if inner dimensions are known at compile time
        bool innerDimsKnown{true};
        int knownInnerDims[3] = {1,1,1};
        forStatement *innerSmnt = getInnerMostInnerLoop(*outerMostOuterLoop);
        statementArray path = oklForStatement::getOklLoopPath(*innerSmnt);

        int innerIndex;
        const int pathCount = (int) path.length();
        for (int i = 0; i < pathCount; ++i) {
          forStatement &pathSmnt = *((forStatement*) path[i]);
          oklForStatement oklForSmnt(pathSmnt);

          if(pathSmnt.hasAttribute("inner")) {
            innerIndex = oklForSmnt.oklLoopIndex();
            if(oklForSmnt.getIterationCount()->canEvaluate()) {
              knownInnerDims[innerIndex] = (int) oklForSmnt.getIterationCount()->evaluate();
            } else {
              std::string s = oklForSmnt.getIterationCount()->toString();
              if(s.find("_occa_tiled_") != std::string::npos) {
                size_t tile_size = s.find_first_of("123456789");
                OCCA_ERROR("@tile size is undefined!",tile_size != std::string::npos);
                knownInnerDims[innerIndex] = std::stoi(s.substr(tile_size));
              } else {
                //loop bounds are unknown at compile time
                innerDimsKnown=false;
                break;
              }
            }
          }
        }
        const int knownInnerDim =  knownInnerDims[0]
                                 * knownInnerDims[1]
                                 * knownInnerDims[2];
        const int maxInnerDim =  maxInnerDims[0]
                               * maxInnerDims[1]
                               * maxInnerDims[2];

        if (innerDimsKnown & maxInnerDimsKnown) {
          if (knownInnerDim > maxInnerDim) {
            outerMostOuterLoop->printError("[@inner] loop dimensions larger then allowed by [@max_inner_dims]");
            success=false;
            return;
          }
        }

        // Determine how long the exclusive array should be
        int exclusiveArraySize = 1024;
        if (maxInnerDimsKnown) {
          exclusiveArraySize = maxInnerDim;
        }
        if (innerDimsKnown) {
          exclusiveArraySize = knownInnerDim;
        }

        // Make exclusive variable declaration into an array
        // For example:
        //    const int x
        // -> const int x[1024]
        operatorToken startToken(var.source->origin,
                                 op::bracketStart);
        operatorToken endToken(var.source->origin,
                               op::bracketEnd);
        // Add exclusive array to the beginning
        var.vartype.arrays.insert(
          var.vartype.arrays.begin(),
          array_t(startToken,
                  endToken,
                  new primitiveNode(var.source,
                                    exclusiveArraySize))
        );
      }

      exprNode* serialParser::addExclusiveVariableArrayAccessor(statement_t &smnt,
                                                                exprNode &expr,
                                                                variable_t &var) {
        // Add the array access to the variable
        // For example:
        //    x
        // -> x[exclusive_index]
        keyword_t &keyword = smnt.getScopeKeyword(exclusiveIndexName);
        variable_t &indexVar = ((variableKeyword&) keyword).variable;

        variableNode indexVarNode(var.source,
                                  indexVar);

        return new subscriptNode(var.source,
                                 expr,
                                 indexVarNode);
      }
    }
  }
}
