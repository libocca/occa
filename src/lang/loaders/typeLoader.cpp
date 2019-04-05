#include <occa/lang/builtins/types.hpp>
#include <occa/lang/loaders/typeLoader.hpp>
#include <occa/lang/statementContext.hpp>
#include <occa/lang/token.hpp>
#include <occa/lang/tokenContext.hpp>
#include <occa/lang/variable.hpp>

namespace occa {
  namespace lang {
    typeLoader_t::typeLoader_t(tokenContext_t &tokenContext_,
                               statementContext_t &smntContext_,
                               const keywords_t &keywords_) :
      tokenContext(tokenContext_),
      smntContext(smntContext_),
      keywords(keywords_),
      success(true) {}

    bool typeLoader_t::loadType(vartype_t &vartype) {
      // TODO: Handle weird () cases:
      //        int (*const (*const a))      -> int * const * const a;
      //        int (*const (*const (*a)))() -> int (* const * const *a)();
      // Set the name in loadBaseType and look for (*)() or (^)()
      //   to stop qualifier merging
      loadBaseType(vartype);
      if (!success || !vartype.isValid()) {
        return false;
      }

      setVartypePointers(vartype);
      if (!success) {
        return false;
      }

      setVartypeReference(vartype);
      return true;
    }

    bool typeLoader_t::loadBaseType(vartype_t &vartype) {
      // Type was already loaded
      if (vartype.type) {
        return true;
      }

      const int tokenCount = tokenContext.size();
      int tokenPos;

      bool typedefing = false;
      for (tokenPos = 0; tokenPos < tokenCount; ++tokenPos) {
        token_t *token = tokenContext[tokenPos];

        keyword_t &keyword = keywords.get(smntContext, token);
        const int kType    = keyword.type();
        if (kType & keywordType::none) {
          break;
        }

        if (kType & keywordType::qualifier) {
          const qualifier_t &qualifier = keyword.to<qualifierKeyword>().qualifier;
          type_t *type = NULL;
          if (qualifier == typedef_) {
            typedefing = true;
          } else if (qualifier == struct_) {
            if (typedefing) {
              token->printError("Typedef'd structs are not supported yet");
              success = false;
            }
          } else if (qualifier == enum_) {
            // TODO: type = loadEnum();
            token->printError("Enums are not supported yet");
            success = false;
          } else if (qualifier == union_) {
            // TODO: type = loadUnion();
            token->printError("Enums are not supported yet");
            success = false;
          } else if (qualifier == class_) {
            // TODO: type = loadClass();
            token->printError("Enums are not supported yet");
            success = false;
          }
          if (!success) {
            return false;
          }
          if (!type) {
            loadVartypeQualifier(token,
                                 keyword.to<qualifierKeyword>().qualifier,
                                 vartype);
          } else {
            vartype.type = type;
            vartype.typeToken = (identifierToken*) type->source->clone();
          }
          continue;
        }
        if ((kType & keywordType::type) &&
            !vartype.isValid()) {
          vartype.type = &(keyword.to<typeKeyword>().type_);
          vartype.typeToken = (identifierToken*) token->clone();
          continue;
        }

        break;
      }

      if (tokenPos) {
        tokenContext += tokenPos;
      } else {
        tokenContext.printError("Unable to load type");
        return false;
      }

      if (vartype.isValid()) {
        return true;
      }

      if (vartype.has(long_) ||
          vartype.has(longlong_)) {
        vartype.type = &int_;
        return true;
      }

      tokenContext.printError("Expected a type");
      return false;
    }

    void typeLoader_t::loadVartypeQualifier(token_t *token,
                                            const qualifier_t &qualifier,
                                            vartype_t &vartype) {
      // Handle long/long long case
      if (&qualifier == &long_) {
        if (vartype.has(long_)) {
          vartype -= long_;
          vartype.add(token->origin,
                      longlong_);
        }
        else if (vartype.has(longlong_)) {
          token->printWarning("'long long long' is tooooooo long,"
                              " ignoring additional longs");
        }
        else {
          vartype.add(token->origin,
                      long_);
        }
        return;
      }

      // Non-long qualifiers
      if (!vartype.has(qualifier)) {
        vartype.add(token->origin,
                    qualifier);
      } else {
        token->printWarning("Ignoring duplicate qualifier");
      }
    }

    void typeLoader_t::setVartypePointers(vartype_t &vartype) {
      while (success && tokenContext.size()) {
        if (!(token_t::safeOperatorType(tokenContext[0]) & operatorType::mult)) {
          break;
        }
        ++tokenContext;
        setVartypePointer(vartype);
      }
    }

    void typeLoader_t::setVartypePointer(vartype_t &vartype) {
      pointer_t pointer;

      const int tokens = tokenContext.size();
      int tokenPos;
      for (tokenPos = 0; tokenPos < tokens; ++tokenPos) {
        token_t *token     = tokenContext[tokenPos];
        keyword_t &keyword = keywords.get(smntContext, token);
        if (!(keyword.type() & keywordType::qualifier)) {
          break;
        }

        const qualifier_t &qualifier = keyword.to<qualifierKeyword>().qualifier;
        if (!(qualifier.type() & qualifierType::forPointers)) {
          token->printError("Cannot add this qualifier to a pointer");
          success = false;
          break;
        }
        pointer.add(token->origin,
                    qualifier);
      }

      tokenContext += tokenPos;

      if (success) {
        vartype += pointer;
      }
    }

    void typeLoader_t::setVartypeReference(vartype_t &vartype) {
      if (!tokenContext.size()) {
        return;
      }
      if (!(token_t::safeOperatorType(tokenContext[0]) & operatorType::bitAnd)) {
        return;
      }
      vartype.setReferenceToken(tokenContext[0]);
      ++tokenContext;
    }

    bool loadType(tokenContext_t &tokenContext,
                  statementContext_t &smntContext,
                  const keywords_t &keywords,
                  vartype_t &vartype) {
      typeLoader_t loader(tokenContext, smntContext, keywords);
      return loader.loadType(vartype);
    }

    bool loadBaseType(tokenContext_t &tokenContext,
                      statementContext_t &smntContext,
                      const keywords_t &keywords,
                      vartype_t &vartype) {
      typeLoader_t loader(tokenContext, smntContext, keywords);
      return loader.loadBaseType(vartype);
    }

    bool isLoadingStruct(tokenContext_t &tokenContext,
                         statementContext_t &smntContext,
                         const keywords_t &keywords) {
      tokenContext.push();
      tokenContext.supressErrors = true;

      vartype_t vartype;
      loadType(tokenContext, smntContext, keywords, vartype);

      tokenContext.supressErrors = false;
      tokenContext.pop();

      return (!vartype.isValid()   &&  // Should not have a base type since we're defining it
              vartype.has(struct_) &&  // Should have struct_
              !vartype.has(typedef_)); // typedef struct is not loaded as a struct
    }
  }
}
