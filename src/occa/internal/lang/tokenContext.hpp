#ifndef OCCA_INTERNAL_LANG_TOKENCONTEXT_HEADER
#define OCCA_INTERNAL_LANG_TOKENCONTEXT_HEADER

#include <list>
#include <map>
#include <vector>

#include <occa/types.hpp>

namespace occa {
  namespace lang {
    class exprNode;
    class identifierToken;
    class keywords_t;
    class parser_t;
    class statementContext_t;
    class tokenRange;
    class token_t;

    typedef bitfield opType_t;

    typedef std::vector<token_t*>   tokenVector;
    typedef std::list<tokenRange>   tokenRangeList;
    typedef std::vector<tokenRange> tokenRangeVector;
    typedef std::map<int, int>      intIntMap;

    class tokenRange {
    public:
      int start, end;

      tokenRange();

      tokenRange(const int start_,
                 const int end_);
    };

    class tokenContext_t {
    public:
      tokenVector tokens;
      // Keep track of used tokens (e.g. not commnents)
      intVector tokenIndices;

      intIntMap pairs;
      intVector semicolons;
      bool hasError;
      bool supressErrors;

      tokenRangeList stack;
      tokenRange tp;

      tokenContext_t();
      ~tokenContext_t();

      void clear();
      void setup(const tokenVector &tokens_);
      void setupTokenIndices();

      void findPairs();
      void findSemicolons();

      bool indexInRange(const int index) const;

      void set(const int start);
      void set(const int start,
               const int end);
      void set(const tokenRange &range);

      void push();
      void push(const int start);
      void push(const int start,
                const int end);
      void push(const tokenRange &range);

      void pushPairRange();

      tokenRange pop();
      void popAndSkip();

      int position() const;

      int size() const;

      void getSkippedTokens(tokenVector &skippedTokens,
                            const int start,
                            const int end);

      token_t* getToken(const int index);
      token_t* getNativeToken(const int index);

      void setToken(const int index,
                    token_t *value);

      token_t* operator [] (const int index);
      tokenContext_t& operator ++ ();
      tokenContext_t& operator ++ (int);
      tokenContext_t& operator += (const int offset);

      token_t* end();

      token_t* getPrintToken(const bool atEnd);

      void printWarning(const std::string &message);
      void printWarningAtEnd(const std::string &message);

      void printError(const std::string &message);
      void printErrorAtEnd(const std::string &message);

      void getTokens(tokenVector &tokens_);
      void getAndCloneTokens(tokenVector &tokens_);

      int getClosingPair();
      token_t* getClosingPairToken();

      int getNextOperator(const opType_t &opType);

      exprNode* parseExpression(statementContext_t &smntContext,
                                parser_t &parser);
      exprNode* parseExpression(statementContext_t &smntContext,
                                parser_t &parser,
                                const int start,
                                const int end);

      token_t* replaceIdentifier(statementContext_t &smntContext,
                                 const keywords_t &keywords,
                                 identifierToken &identifier);

      void debugPrint();
    };
  }
}
#endif
