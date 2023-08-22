#include <occa/internal/lang/expr/exprNode.hpp>
#include <occa/internal/lang/loaders/enumLoader.hpp>
#include <occa/internal/lang/parser.hpp>
#include <occa/internal/lang/token.hpp>
#include <occa/internal/lang/type/enum.hpp>

namespace occa {
  namespace lang {
    enumLoader_t::enumLoader_t(tokenContext_t &tokenContext_,
                                statementContext_t &smntContext_,
                                parser_t &parser_
                                ) :
      tokenContext(tokenContext_),
      smntContext(smntContext_),
      parser(parser_) {}

    bool enumLoader_t::loadEnum(enum_t *&type) {
      type = NULL;

      // Store type declarations in temporary block statement
      blockStatement *blockSmnt = new blockStatement(smntContext.up,
                                                     tokenContext[0]);
      smntContext.pushUp(*blockSmnt);

      identifierToken *nameToken = NULL;
      const bool hasName = token_t::safeType(tokenContext[0]) & tokenType::identifier;
      if (hasName) {
        nameToken = (identifierToken*) tokenContext[0];
        ++tokenContext;
      }

      opType_t opType = token_t::safeOperatorType(tokenContext[0]);
      if (!(opType & (operatorType::braceStart |
                      operatorType::scope))) {
        tokenContext.printError("Expected enum body {}");
        delete blockSmnt;
        smntContext.popUp();
        return false;
      }
      tokenContext.pushPairRange();
      // Load type expression statements
      enumeratorVector enumerators;
      if (tokenContext.size()) {
        identifierToken &source = (tokenContext[0]->clone()->to<identifierToken>());
        while (source.value != "") {
          exprNode *expr_ = NULL;
          ++tokenContext;
          if ((token_t::safeOperatorType(tokenContext[0]) & operatorType::assign)) {
            ++tokenContext;
            const int end = tokenContext.getNextOperator(operatorType::comma);
            if (end>0) {
              expr_ = parser.parseTokenContextExpression(0, end);
              tokenContext += end;
            } else {
              expr_ = parser.parseTokenContextExpression();
            }
          }
          enumerators.push_back(*(new enumerator_t(&source, expr_)));
          if (!(token_t::safeOperatorType(tokenContext[0]) & operatorType::comma)) {
            break;
          }
          ++tokenContext;
          source = (tokenContext[0]->clone()->to<identifierToken>());
        }
      }
      delete blockSmnt;
      smntContext.popUp();
      tokenContext.popAndSkip();

      type = nameToken ? new enum_t(*nameToken) : new enum_t();
      type->addEnumerators(enumerators);

      return true;
    }

    bool loadEnum(tokenContext_t &tokenContext,
                    statementContext_t &smntContext,
                    parser_t &parser,
                    enum_t *&type) {
      enumLoader_t loader(tokenContext, smntContext, parser);
      return loader.loadEnum(type);
    }
  }
}
