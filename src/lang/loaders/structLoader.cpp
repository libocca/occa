#include <occa/internal/lang/expr/exprNode.hpp>
#include <occa/internal/lang/loaders/structLoader.hpp>
#include <occa/internal/lang/parser.hpp>
#include <occa/internal/lang/statement/declarationStatement.hpp>
#include <occa/internal/lang/token.hpp>
#include <occa/internal/lang/type/struct.hpp>
#include <occa/internal/lang/variable.hpp>

namespace occa {
  namespace lang {
    structLoader_t::structLoader_t(tokenContext_t &tokenContext_,
                                   statementContext_t &smntContext_,
                                   parser_t &parser_) :
      tokenContext(tokenContext_),
      smntContext(smntContext_),
      parser(parser_) {}

    bool structLoader_t::loadStruct(struct_t *&type) {
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
        tokenContext.printError("Expected struct body {}");
        delete blockSmnt;
        smntContext.popUp();
        return false;
      }

      tokenContext.pushPairRange();

      // Load type declaration statements
      statement_t *smnt = parser.getNextStatement();
      variableVector fields;
      while (smnt) {
        const int sType = smnt->type();
        if (!(sType & statementType::declaration)) {
          if (sType & (statementType::function |
                       statementType::functionDecl)) {
            smnt->printError("Struct functions are not supported yet");
          } else if (sType & statementType::classAccess) {
            smnt->printError("Access modifiers are not supported yet");
          } else {
            smnt->printError("Expected variable declaration statements");
          }
          delete blockSmnt;
          smntContext.popUp();
          return false;
        }

        variableDeclarationVector &declarations = (smnt
                                                   ->to<declarationStatement>()
                                                   .declarations);
        const int varCount = (int) declarations.size();
        for (int i = 0; i < varCount; ++i) {
          variableDeclaration &decl = declarations[i];
          if (decl.value) {
            decl.value->printError("Struct fields cannot have default values");
            delete blockSmnt;
            smntContext.popUp();
            return false;
          }
          fields.push_back(decl.variable());
        }
        delete smnt;
        smnt = parser.getNextStatement();
      }

      delete blockSmnt;
      smntContext.popUp();

      tokenContext.popAndSkip();

      type = nameToken ? new struct_t(*nameToken) : new struct_t();
      type->addFields(fields);

      return true;
    }

    bool loadStruct(tokenContext_t &tokenContext,
                    statementContext_t &smntContext,
                    parser_t &parser,
                    struct_t *&type) {
      structLoader_t loader(tokenContext, smntContext, parser);
      return loader.loadStruct(type);
    }
  }
}
