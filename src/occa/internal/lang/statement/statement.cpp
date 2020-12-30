#include <occa/internal/lang/statement/statement.hpp>
#include <occa/internal/lang/statement/blockStatement.hpp>
#include <occa/internal/lang/statement/declarationStatement.hpp>
#include <occa/internal/lang/variable.hpp>
#include <occa/internal/lang/token.hpp>
#include <occa/internal/lang/expr.hpp>

namespace occa {
  namespace lang {
    namespace statementType {
      const int none         = (1 << 0);
      const int all          = -1;

      const int empty        = (1 << 1);

      const int directive    = (1 << 2);
      const int pragma       = (1 << 3);
      const int comment      = (1 << 4);

      const int block        = (1 << 5);
      const int namespace_   = (1 << 6);

      const int function     = (1 << 7);
      const int functionDecl = (1 << 8);

      const int class_       = (1 << 9);
      const int classAccess  = (1 << 10);

      const int enum_        = (1 << 11);
      const int union_       = (1 << 12);

      const int expression   = (1 << 13);
      const int declaration  = (1 << 14);

      const int goto_        = (1 << 15);
      const int gotoLabel    = (1 << 16);

      const int if_          = (1 << 17);
      const int elif_        = (1 << 18);
      const int else_        = (1 << 19);
      const int for_         = (1 << 20);
      const int while_       = (1 << 21);
      const int switch_      = (1 << 22);
      const int case_        = (1 << 23);
      const int default_     = (1 << 24);
      const int continue_    = (1 << 25);
      const int break_       = (1 << 26);

      const int return_      = (1 << 27);

      const int attribute    = (1 << 28);

      const int sourceCode   = (1 << 29);

      const int blockStatements = (
        block        |
        elif_        |
        else_        |
        for_         |
        functionDecl |
        if_          |
        namespace_   |
        switch_      |
        while_
      );
    }

    statement_t::statement_t(blockStatement *up_,
                             const token_t *source_) :
      up(up_),
      source(token_t::clone(source_)),
      attributes() {}

    statement_t::statement_t(blockStatement *up_,
                             const statement_t &other) :
      up(up_),
      source(token_t::clone(other.source)),
      attributes() {}

    statement_t::~statement_t() {
      delete source;
    }

    statement_t& statement_t::clone(blockStatement *up_) const {
      statement_t &s = clone_(up_);
      s.attributes = attributes;
      return s;
    }

    statement_t* statement_t::clone(blockStatement *up_,
                                    statement_t *smnt) {
      if (smnt) {
        return &(smnt->clone(up_));
      }
      return NULL;
    }

    void statement_t::swapSource(statement_t &other) {
      token_t *prevSource = source;
      source = other.source;
      other.source = prevSource;
    }

    bool statement_t::hasInScope(const std::string &name) {
      if (up) {
        return up->hasInScope(name);
      }
      return false;
    }

    keyword_t& statement_t::getScopeKeyword(const std::string &name) {
      return up->getScopeKeyword(name);
    }

    void statement_t::addAttribute(const attributeToken_t &attribute) {
      attributes[attribute.name()] = attribute;
    }

    bool statement_t::hasAttribute(const std::string &attr) const {
      return (attributes.find(attr) != attributes.end());
    }

    std::string statement_t::toString() const {
      printer pout;
      pout << (*this);
      return pout.str();
    }

    statement_t::operator std::string() const {
      return toString();
    }

    int statement_t::childIndex() const {
      if (!up ||
          !up->is<blockStatement>()) {
        return -1;
      }
      blockStatement &upBlock = *((blockStatement*) up);
      const int childrenCount = (int) upBlock.children.length();
      for (int i = 0; i < childrenCount; ++i) {
        if (upBlock.children[i] == this) {
          return i;
        }
      }
      return -1;
    }

    void statement_t::removeFromParent() {
      if (up) {
        up->remove(*this);
      }
    }

    void statement_t::replaceWith(statement_t &other) {
      if (!up) {
        return;
      }

      up->addBefore(*this, other);
      up->remove(*this);
      up = NULL;
    }

    statementArray statement_t::getParentPath() {
      statementArray arr;

      statement_t *smnt = up;
      while (smnt) {
        arr.push(smnt);
        smnt = smnt->up;
      }

      return arr.inplaceReverse();
    }

    statementArray statement_t::getInnerStatements() {
      return statementArray();
    }

    exprNodeArray statement_t::getExprNodes() {
      exprNodeArray nodes;

      for (smntExprNode &smntExpr : getDirectExprNodes()) {
        for (exprNode *childNode : smntExpr.node->getNestedChildren()) {
          nodes.push({this, childNode});
        }
        nodes.push(smntExpr);
      }

      return nodes;
    }

    exprNodeArray statement_t::getDirectExprNodes() {
      return exprNodeArray();
    }

    void statement_t::replaceExprNode(exprNode *currentNode, exprNode *newNode) {
      if (currentNode != newNode) {
        safeReplaceExprNode(currentNode, newNode);
      }
    }

    void statement_t::safeReplaceExprNode(exprNode *currentNode, exprNode *newNode) {}

    void statement_t::replaceKeyword(const keyword_t &currentKeyword,
                                     keyword_t &newKeyword) {
      const int kType = currentKeyword.type();

      if (kType & keywordType::variable) {
        const variable_t &currentVar = ((const variableKeyword&) currentKeyword).variable;
        variable_t &newVar = ((variableKeyword&) newKeyword).variable;

        replaceVariable(currentVar, newVar);
      }
      else if (kType & keywordType::function) {
        const function_t &currentFunc = ((const functionKeyword&) currentKeyword).function;
        function_t &newFunc = ((functionKeyword&) newKeyword).function;

        replaceFunction(currentFunc, newFunc);
      }
      else if (kType & keywordType::type) {
        const type_t &currentType = ((const typeKeyword&) currentKeyword).type_;
        type_t &newType = ((typeKeyword&) newKeyword).type_;

        replaceType(currentType, newType);
      }
    }

    void statement_t::replaceVariable(const variable_t &currentVar, variable_t &newVar) {
      statementArray::from(*this)
          .flatFilterByExprType(exprNodeType::variable)
          .inplaceMap([&](smntExprNode smntExpr) -> exprNode* {
              variableNode *varNode = (variableNode*) smntExpr.node;
              variable_t &var = ((variableNode*) smntExpr.node)->value;

              if (&var != &currentVar) {
                return varNode;
              }

              return new variableNode(varNode->token, newVar);
            });
    }

    void statement_t::replaceFunction(const function_t &currentFunc, function_t &newFunc) {
      statementArray::from(*this)
          .flatFilterByExprType(exprNodeType::function)
          .inplaceMap([&](smntExprNode smntExpr) -> exprNode* {
              functionNode *funcNode = (functionNode*) smntExpr.node;
              function_t &func = ((functionNode*) smntExpr.node)->value;

              if (&func != &currentFunc) {
                return funcNode;
              }

              return new functionNode(funcNode->token, newFunc);
            });
    }

    void statement_t::replaceType(const type_t &currentType, type_t &newType) {
      statementArray::from(*this)
          .flatFilterByExprType(exprNodeType::type)
          .inplaceMap([&](smntExprNode smntExpr) -> exprNode* {
              typeNode *_typeNode = (typeNode*) smntExpr.node;
              type_t &type = ((typeNode*) smntExpr.node)->value;

              if (&type != &currentType) {
                return _typeNode;
              }

              return new typeNode(_typeNode->token, newType);
            });
    }

    void statement_t::updateVariableReferences() {
      std::map<variable_t*, variable_t*> variablesToReplace;

      statementArray::from(*this)
          .flatFilterByExprType(exprNodeType::variable)
          .forEach([&](smntExprNode smntExpr) {
              statement_t *smnt = smntExpr.smnt;
              variable_t &var = ((variableNode*) smntExpr.node)->value;
              const std::string &name = var.name();

              if (!name.size()) {
                return;
              }

              // No need to replace the variable defined in the statement
              if ((smnt->type() & statementType::declaration)
                  && ((declarationStatement*) smnt)->declaresVariable(var)) {
                return;
              }

              keyword_t &keyword = smntExpr.smnt->getScopeKeyword(name);
              if (!(keyword.type() & keywordType::variable)) {
                smntExpr.node->printError("Variable not defined in this scope");
                return;
              }

              variable_t &scopedVar = keyword.to<variableKeyword>().variable;

              if (&scopedVar != &var) {
                variablesToReplace.insert({&var, &scopedVar});
              }
          });

      for (auto it : variablesToReplace) {
        replaceVariable(*it.first, *it.second);
      }
    }

    void statement_t::updateIdentifierReferences() {
      exprNodeArray arr = (
        statementArray::from(*this)
        .flatFilterByExprType(exprNodeType::identifier)
      );

      updateIdentifierReferences(arr);
    }

    void statement_t::updateIdentifierReferences(exprNode *expr) {
      exprNodeArray arr = (
        exprNodeArray::from(this, expr)
        .flatFilterByExprType(exprNodeType::identifier)
      );

      updateIdentifierReferences(arr);
    }

    void statement_t::updateIdentifierReferences(exprNodeArray &arr) {
      arr.inplaceMap([&](smntExprNode smntExpr) -> exprNode* {
          statement_t *smnt = smntExpr.smnt;
          identifierNode &node = (identifierNode&) *smntExpr.node;

          const std::string &name = node.value;

          keyword_t &keyword = smnt->getScopeKeyword(name);
          const int kType = keyword.type();
          if (!(kType & (keywordType::type     |
                         keywordType::variable |
                         keywordType::function))) {
            return &node;
          }

          if (kType & keywordType::variable) {
            return (
              new variableNode(node.token,
                               ((variableKeyword&) keyword).variable)
            );
          }
          if (kType & keywordType::function) {
            return (
              new functionNode(node.token,
                               ((functionKeyword&) keyword).function)
            );
          }
          // keywordType::type
          return (
            new typeNode(node.token,
                         ((typeKeyword&) keyword).type_)
          );
        });
    }

    void statement_t::debugPrint() const {
      io::stdout << toString();
    }

    void statement_t::printWarning(const std::string &message) const {
      source->printWarning(message);
    }

    void statement_t::printError(const std::string &message) const {
      source->printError(message);
    }

    printer& operator << (printer &pout,
                          const statement_t &smnt) {
      smnt.print(pout);
      return pout;
    }
  }
}
