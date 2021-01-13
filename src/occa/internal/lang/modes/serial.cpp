#include <set>

#include <occa/internal/lang/modes/serial.hpp>
#include <occa/internal/lang/modes/okl.hpp>
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
              defineExclusiveVariableAsArray(var);
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

      void serialParser::defineExclusiveVariableAsArray(variable_t &var) {
        // TODO: Dynamic array sizes
        // Define the variable as a stack array
        // For example:
        //    const int x
        // -> const int x[256]
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
                                    256))
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
