#include <occa/lang/expr.hpp>
#include <occa/lang/keyword.hpp>
#include <occa/lang/statement.hpp>
#include <occa/lang/transforms/builtins/fillExprIdentifiers.hpp>

namespace occa {
  namespace lang {
    namespace transforms {
      fillExprIdentifiers_t::fillExprIdentifiers_t(blockStatement *scopeSmnt_) :
        scopeSmnt(scopeSmnt_) {
        validExprNodeTypes = exprNodeType::identifier;
      }

      exprNode* fillExprIdentifiers_t::transformExprNode(exprNode &node) {
        if (!scopeSmnt) {
          return &node;
        }
        const std::string &name = ((identifierNode&) node).value;
        keyword_t &keyword = scopeSmnt->getScopeKeyword(name);
        const int kType = keyword.type();
        if (!(kType & (keywordType::type     |
                       keywordType::variable |
                       keywordType::function))) {
          return &node;
        }

        if (kType & keywordType::variable) {
          return new variableNode(node.token,
                                  ((variableKeyword&) keyword).variable);
        }
        if (kType & keywordType::function) {
          return new functionNode(node.token,
                                  ((functionKeyword&) keyword).function);
        }
        // keywordType::type
        return new typeNode(node.token,
                            ((typeKeyword&) keyword).type_);
      }
    }
  }
}
