#ifndef OCCA_LANG_TOKENCONTEXT_HEADER
#define OCCA_LANG_TOKENCONTEXT_HEADER

#include <list>
#include <map>
#include <vector>

#include <occa/types.hpp>

namespace occa {
  namespace lang {
    class tokenRange;
    class token_t;
    class exprNode;

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

    class tokenContext {
    public:
      tokenVector tokens;
      intIntMap pairs;
      intVector semicolons;
      bool hasError;

      tokenRangeList stack;
      tokenRange tp;

      tokenContext();
      ~tokenContext();

      void clear();
      void setup();

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

      void pushPairRange(const int pairStart);

      tokenRange pop();
      void popAndSkip();

      int position() const;
      int size() const;
      token_t* operator [] (const int index);
      void setToken(const int index,
                    token_t *value);

      token_t* end();

      token_t* getPrintToken(const bool atEnd);

      void printWarning(const std::string &message);
      void printWarningAtEnd(const std::string &message);

      void printError(const std::string &message);
      void printErrorAtEnd(const std::string &message);

      void getTokens(tokenVector &tokens_);
      void getAndCloneTokens(tokenVector &tokens_);

      int getClosingPair(const int index);
      token_t* getClosingPairToken(const int index);

      int getNextOperator(const opType_t &opType);

      void debugPrint();
    };
  }
}
#endif
