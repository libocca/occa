#include <occa/lang/modes/serial.hpp>
#include <occa/lang/modes/okl.hpp>
#include <occa/lang/builtins/types.hpp>

namespace occa {
  namespace lang {
    namespace okl {
      const std::string serialParser::exclusiveIndexName = "_occa_exclusive_index";

      serialParser::serialParser(const occa::properties &settings_) :
        parser_t(settings_) {

        okl::addAttributes(*this);
      }

      void serialParser::onClear() {}

      void serialParser::afterParsing() {
        if (!success) return;
        if (settings.get("okl/validate", true)) {
          success = checkKernels(root);
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

        // Get @kernels
        statementPtrVector kernelSmnts;
        findStatementsByAttr(statementType::functionDecl,
                             "kernel",
                             root,
                             kernelSmnts);
        const int kernels = (int) kernelSmnts.size();
        for (int i = 0; i < kernels; ++i) {
          setupKernel(*((functionDeclStatement*) kernelSmnts[i]));
          if (!success) {
            break;
          }
        }
      }

      void serialParser::setupKernel(functionDeclStatement &kernelSmnt) {
        // @kernel -> extern "C"
        function_t &func = kernelSmnt.function;
        attributeToken_t &kernelAttr = kernelSmnt.attributes["kernel"];
        qualifiers_t &qualifiers = func.returnType.qualifiers;
        // Add extern "C" [__declspec(dllexport)]
#if OCCA_OS == OCCA_WINDOWS_OS
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
        statementExprMap exprMap;
        findStatements(exprNodeType::variable,
                       root,
                       exclusiveVariableMatcher,
                       exprMap);

        setupExclusiveDeclarations(exprMap);
        if (!success) return;
        setupExclusiveIndices();
        if (!success) return;
        transformExprNodes(exprNodeType::variable,
                           root,
                           updateExclusiveExprNodes);
      }

      void serialParser::setupExclusiveDeclarations(statementExprMap &exprMap) {
        statementExprMap::iterator it = exprMap.begin();
        while (it != exprMap.end()) {
          statement_t *smnt = it->first;
          if (smnt->type() == statementType::declaration) {
            setupExclusiveDeclaration(*((declarationStatement*) smnt));
          }
          if (!success) {
            break;
          }
          ++it;
        }
      }

      void serialParser::setupExclusiveDeclaration(declarationStatement &declSmnt) {
        // Make sure the @exclusive variable is declared here, not used
        if (!exclusiveIsDeclared(declSmnt)) {
          return;
        }

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
            keyword.printError(exclusiveIndexName
                               + " is a restricted OCCA keyword");
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

      bool serialParser::exclusiveIsDeclared(declarationStatement &declSmnt) {
        const int declCount = (int) declSmnt.declarations.size();
        for (int i = 0; i < declCount; ++i) {
          variableDeclaration &decl = declSmnt.declarations[i];
          if (decl.variable->hasAttribute("exclusive")) {
            return true;
          }
        }
        return false;
      }

      void serialParser::setupExclusiveIndices() {
        transforms::smntTreeNode innerRoot;
        findStatementTree(statementType::for_,
                          root,
                          exclusiveInnerLoopMatcher,
                          innerRoot);

        // Get outer-most inner loops
        statementPtrVector outerMostInnerLoops;
        const int childCount = (int) innerRoot.size();
        for (int i = 0; i < childCount; ++i) {
          outerMostInnerLoops.push_back(innerRoot[i]->smnt);
        }
        // Get inner-most inner loops
        statementPtrVector innerMostInnerLoops;
        getInnerMostLoops(innerRoot, innerMostInnerLoops);

        // Initialize index variable to 0 before outer-most inner loop
        const int outerMostInnerCount = (int) outerMostInnerLoops.size();
        for (int i = 0; i < outerMostInnerCount; ++i) {
          forStatement &innerSmnt = *((forStatement*) outerMostInnerLoops[i]);
          keyword_t &keyword = innerSmnt.getScopeKeyword(exclusiveIndexName);
          variable_t &indexVar = ((variableKeyword&) keyword).variable;

          variableNode indexVarNode(innerSmnt.source,
                                    indexVar);
          primitiveNode zeroNode(innerSmnt.source,
                                 0);
          binaryOpNode assign(innerSmnt.source,
                              op::assign,
                              indexVarNode,
                              zeroNode);

          innerSmnt.up->addBefore(
            innerSmnt,
            *(new expressionStatement(&innerSmnt,
                                      *(assign.clone())))
          );
        }

        // Update index after inner-most inner loop
        const int innerMostInnerCount = (int) innerMostInnerLoops.size();
        for (int i = 0; i < innerMostInnerCount; ++i) {
          forStatement &innerSmnt = *((forStatement*) innerMostInnerLoops[i]);
          keyword_t &keyword = innerSmnt.getScopeKeyword(exclusiveIndexName);
          variable_t &indexVar = ((variableKeyword&) keyword).variable;

          variableNode indexVarNode(innerSmnt.source,
                                    indexVar);
          leftUnaryOpNode increment(innerSmnt.source,
                                    op::leftIncrement,
                                    indexVarNode);
          innerSmnt.addLast(
            *(new expressionStatement(&innerSmnt,
                                      *(increment.clone())))
          );
        }
      }

      bool serialParser::exclusiveVariableMatcher(exprNode &expr) {
        return expr.hasAttribute("exclusive");
      }

      bool serialParser::exclusiveInnerLoopMatcher(statement_t &smnt) {
        return (
          smnt.hasAttribute("inner")
          && smnt.hasInScope(exclusiveIndexName)
        );
      }

      void serialParser::getInnerMostLoops(transforms::smntTreeNode &innerRoot,
                                           statementPtrVector &loopSmnts) {
        const int count = (int) innerRoot.size();
        if (!count) {
          if (innerRoot.smnt) {
            loopSmnts.push_back(innerRoot.smnt);
          }
          return;
        }
        for (int i = 0; i < count; ++i) {
          getInnerMostLoops(*(innerRoot.children[i]),
                            loopSmnts);
        }
      }

      exprNode* serialParser::updateExclusiveExprNodes(statement_t &smnt,
                                                       exprNode &expr,
                                                       const bool isBeingDeclared) {
        variable_t &var = ((variableNode&) expr).value;
        if (!var.hasAttribute("exclusive")) {
          return &expr;
        }

        if (isBeingDeclared) {
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
          return &expr;
        }

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
