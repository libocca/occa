#ifndef OCCA_LANG_PREPROCESSOR_HEADER
#define OCCA_LANG_PREPROCESSOR_HEADER

#include <ostream>
#include <vector>
#include <map>
#include <stack>

#include <occa/defines.hpp>
#include <occa/types.hpp>
#include <occa/lang/macro.hpp>
#include <occa/lang/token.hpp>
#include <occa/lang/stream.hpp>

namespace occa {
  namespace lang {
    class tokenizer_t;

    typedef std::map<std::string, bool> stringSet;

    typedef std::vector<token_t*> tokenVector;
    typedef std::stack<token_t*>  tokenStack;
    typedef std::list<token_t*>   tokenList;

    typedef std::map<std::string, macro_t*> macroMap;
    typedef std::map<macro_t*, bool>        macroSet;
    typedef std::vector<macro_t*>           macroVector;
    typedef std::map<token_t*, macroVector> macroEndMap;

    typedef streamMap<token_t*, token_t*> tokenMap;

    namespace ppStatus {
      extern const int reading;
      extern const int ignoring;
      extern const int foundIf;
      extern const int foundElse;
      extern const int finishedIf;
    }

    class preprocessor_t : public withCache<token_t*, token_t*> {
    public:
      typedef void (preprocessor_t::*processDirective_t)(identifierToken &directive);
      typedef std::map<std::string, processDirective_t> directiveMap;

      //---[ Status ]-------------------
      std::vector<int> statusStack;
      int status;

      int passedNewline;
      bool expandingMacros;

      macroSet expandedMacros;
      macroEndMap expandedMacroEnd;
      //================================

      //---[ Macros and Directives ]----
      directiveMap directives;

      macroMap compilerMacros;
      macroMap sourceMacros;
      stringSet standardHeaders;
      //================================

      //---[ Metadata ]-----------------
      occa::properties settings;

      strToBoolMap dependencies;
      int warnings, errors;
      //================================

      //---[ Misc ]---------------------
      tokenizer_t *tokenizer;

      strVector includePaths;
      //================================

      preprocessor_t(const occa::properties &settings_ = occa::properties());
      preprocessor_t(const preprocessor_t &other);
      ~preprocessor_t();

      void init();
      void clear();
      void clear_();

      preprocessor_t& operator = (const preprocessor_t &pp);

      void setSettings(occa::properties settings_);

      void initDirectives();
      void initStandardHeaders();

      void warningOn(token_t *token,
                     const std::string &message);

      void errorOn(token_t *token,
                   const std::string &message);

      virtual tokenMap& clone_() const;

      virtual void* passMessageToInput(const occa::properties &props);

      void pushStatus(const int status_);
      int popStatus();
      void swapReadingStatus();

      void incrementNewline();
      void decrementNewline();

      macro_t* getMacro(const std::string &name);
      macro_t* getSourceMacro();

      token_t* getSourceToken();

      virtual void fetchNext();

      //---[ Public ]-------------------
      void addCompilerDefine(const std::string &name,
                             const std::string &value);

      void removeCompilerDefine(const std::string &name);

      void addSourceDefine(const std::string &name,
                           const std::string &value);

      void removeSourceDefine(const std::string &name);

      strVector getDependencyFilenames() const;
      //================================

      void loadTokenizer();

      bool expandDefinedToken(token_t *token,
                              tokenVector &outputTokens);

      void expandDefinedTokens(tokenVector &inputTokens,
                               tokenVector &outputTokens);

      void expandMacro(identifierToken &source,
                       macro_t &macro);
      void clearExpandedMacros(token_t *token);

      void skipToNewline();
      void getLineTokens(tokenVector &lineTokens);
      void getExpandedLineTokens(tokenVector &lineTokens);
      void warnOnNonEmptyLine(const std::string &message);
      void removeNewline(tokenVector &lineTokens);

      void processToken(token_t *token);

      bool canProcessWhileIgnoring(token_t *token);
      bool processingDirectiveAttribute(operatorToken &opToken,
                                        token_t *&directiveToken);

      void processIdentifier(identifierToken &token);

      void processOperator(operatorToken &opToken);
      void processHashOperator(operatorToken &opToken);
      void processAttributeOperator(operatorToken &opToken);
      void freeAttributeOperatorTokens(token_t &opToken,
                                       token_t &directiveToken,
                                       std::list<token_t*> &prevOutputCache);

      bool lineIsTrue(identifierToken &directive,
                      bool &isTrue);
      bool getIfdef(identifierToken &directive,
                    bool &isTrue);

      void processIf(identifierToken &directive);
      void processIfdef(identifierToken &directive);
      void processIfndef(identifierToken &directive);
      void processElif(identifierToken &directive);
      void processElse(identifierToken &directive);
      void processEndif(identifierToken &directive);

      void processDefine(identifierToken &directive);
      void processUndef(identifierToken &directive);

      void processError(identifierToken &directive);
      void processWarning(identifierToken &directive);

      void processInclude(identifierToken &directive);
      bool isStandardHeader(const std::string &header);
      void processPragma(identifierToken &directive);
      void processOccaPragma(identifierToken &directive,
                             tokenVector &lineTokens);

      int getLineNumber();
      void processLine(identifierToken &directive);
    };
  }
}

#endif
