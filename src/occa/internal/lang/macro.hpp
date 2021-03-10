#ifndef OCCA_INTERNAL_LANG_MACRO_HEADER
#define OCCA_INTERNAL_LANG_MACRO_HEADER

#include <vector>
#include <map>

#include <occa/internal/lang/token.hpp>

namespace occa {
  namespace lang {
    class token_t;
    class identifierToken;
    class macroToken;
    class macro_t;
    class preprocessor_t;

    typedef std::map<std::string, int> intMap;
    typedef intMap::iterator           intMapIterator;

    typedef std::vector<token_t*>    tokenVector;
    typedef std::vector<macroToken*> macroTokenVector;

    //---[ Helper Methods ]-------------
    void freeTokenVectors(std::vector<tokenVector> &tokenVectors);

    void freeTokenVector(macroTokenVector &mTokens);

    void cloneMacroTokenVector(macroTokenVector &newTokens,
                               const macroTokenVector &tokens);
    //==================================

    //---[ Macro Tokens ]---------------
    class macroToken {
    public:
      preprocessor_t &pp;
      token_t *thisToken;

      macroToken(preprocessor_t &pp_,
                 token_t *thisToken_);
      virtual ~macroToken();

      virtual macroToken* clone() const = 0;

      token_t* cloneToken(token_t *token,
                          token_t *sourceToken);

      virtual bool expand(tokenVector &newTokens,
                          token_t *source,
                          std::vector<tokenVector> &args) = 0;

      void expandDefinedTokens(tokenVector &tokens);
    };

    class macroRawToken : public macroToken {
    public:
      macroRawToken(preprocessor_t &pp_,
                    token_t *token_);

      virtual macroToken* clone() const;

      virtual bool expand(tokenVector &newTokens,
                          token_t *source,
                          std::vector<tokenVector> &args);
    };

    class macroArgument : public macroToken {
    public:
      int arg;
      int argc;

      macroArgument(preprocessor_t &pp_,
                    token_t *token_,
                    const int arg_,
                    const int argc_);
      ~macroArgument();

      void expandArg(tokenVector &newTokens,
                     token_t *source,
                     std::vector<tokenVector> &args,
                     const int arg_);

      virtual macroToken* clone() const;

      virtual bool expand(tokenVector &newTokens,
                          token_t *source,
                          std::vector<tokenVector> &args);
    };

    class macroStringify : public macroToken {
    public:
      macroToken *token;

      macroStringify(preprocessor_t &pp_,
                     macroToken *token_);
      ~macroStringify();

      virtual macroToken* clone() const;

      virtual bool expand(tokenVector &newTokens,
                          token_t *source,
                          std::vector<tokenVector> &args);
    };

    class macroConcat : public macroToken {
    public:
      macroTokenVector tokens;

      macroConcat(preprocessor_t &pp_,
                  const macroTokenVector &tokens_);
      ~macroConcat();

      virtual macroToken* clone() const;

      virtual bool expand(tokenVector &newTokens,
                          token_t *source,
                          std::vector<tokenVector> &args);
    };
    //==================================

    //---[ Macro ]----------------------
    class macro_t {
    public:
      static const std::string VA_ARGS;

      preprocessor_t &pp;
      identifierToken &thisToken;
      bool isBuiltin;

      bool isFunctionLike;
      intMap argNames;
      bool hasVarArgs;

      macroTokenVector macroTokens;

      macro_t(preprocessor_t &pp_,
              identifierToken &thisToken_,
              const bool isBuiltin_ = false,
              const bool isFunctionLike = false,
              const bool hasVarArgs = false);

      macro_t(preprocessor_t &pp_,
              const std::string &name_);

      virtual ~macro_t();

      void setupTokenOrigin();

      inline int argCount() const {
        return (((int) argNames.size()) - hasVarArgs);
      }

      inline const std::string& name() const {
        return thisToken.value;
      }

      virtual macro_t& clone(preprocessor_t &pp_) const;

      void loadDefinition();

      void loadFunctionLikeDefinition(tokenVector &tokens);
      bool loadDefinitionArgument(token_t *token);

      void setDefinition(tokenVector &tokens);
      void setMacroTokens(tokenVector &tokens);

      const operator_t* getOperator(macroToken *mToken);
      bool isHash(macroToken *mToken);
      bool isHashhash(macroToken *mToken);

      void stringifyMacroTokens();
      void concatMacroTokens();

      virtual void expand(tokenVector &tokens,
                          identifierToken &source);

      bool loadArgs(identifierToken &source,
                    std::vector<tokenVector> &args);
      bool checkArgs(identifierToken &source,
                     std::vector<tokenVector> &args);

      void errorOn(token_t *token,
                   const std::string &message);
      void errorOn(macroToken *mToken,
                   const std::string &message);

      static macro_t* defineBuiltin(preprocessor_t &pp_,
                                    const std::string &name_,
                                    const std::string &contents);

      static macro_t* define(preprocessor_t &pp_,
                             fileOrigin origin,
                             const std::string &name_,
                             const std::string &contents);
    };
    //==================================
  }
}

#endif
