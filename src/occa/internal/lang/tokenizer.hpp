#ifndef OCCA_INTERNAL_LANG_TOKENIZER_HEADER
#define OCCA_INTERNAL_LANG_TOKENIZER_HEADER

#include <list>
#include <vector>

#include <occa/internal/io.hpp>
#include <occa/internal/utils/trie.hpp>
#include <occa/internal/lang/file.hpp>
#include <occa/internal/lang/printer.hpp>
#include <occa/internal/lang/stream.hpp>

namespace occa {
  namespace lang {
    class token_t;

    typedef std::vector<token_t*>   tokenVector;
    typedef std::list<token_t*>     tokenList;
    typedef std::vector<fileOrigin> originVector;
    typedef trie<const operator_t*> operatorTrie;

    int getEncodingType(const std::string &str);
    int getCharacterEncoding(const std::string &str);
    int getStringEncoding(const std::string &str);

    class tokenizer_t : public baseStream<token_t*> {
    public:
      tokenList outputCache;

      fileOrigin origin;
      filePosition &fp;

      originVector stack;

      operatorTrie operators;
      std::string operatorCharcodes;

      int lastTokenType;
      int lastNonNewlineTokenType;
      int errors, warnings;

      tokenizer_t();

      tokenizer_t(const char *root);

      tokenizer_t(file_t *file_);

      tokenizer_t(fileOrigin origin_);

      tokenizer_t(const tokenizer_t &stream);

      tokenizer_t& operator = (const tokenizer_t &stream);

      virtual ~tokenizer_t();

      void setup();

      virtual baseStream<token_t*>& clone() const;

      virtual void* passMessageToInput(const occa::json &props);

      void set(const char *root);

      void set(file_t *file_);

      void clear();

      void printError(const std::string &message);

      void setLine(const int line);

      bool reachedTheEnd() const;
      virtual bool isEmpty();
      virtual void setNext(token_t *&out);

      void pushSource(const std::string &filename);
      void popSource();

      void push();
      void pop(const bool rewind = false);
      void popAndRewind();
      fileOrigin popTokenOrigin();

      size_t strSize();
      std::string str();

      void countSkippedLines();

      void skipTo(const char delimiter);
      void skipTo(const char *delimiters);
      void skipFrom(const char *delimiters);

      void skipWhitespace();

      int peek();
      int shallowPeek();
      int peekForIdentifier();
      int peekForOperator();

      void getIdentifier(std::string &value);
      bool getString(std::string &value,
                     const int encoding = 0);
      void getRawString(std::string &value);

      token_t* getToken();
      token_t* getIdentifierToken();
      token_t* getPrimitiveToken();
      token_t* getOperatorToken();
      token_t* getLineCommentToken();
      token_t* getBlockCommentToken();
      token_t* getStringToken(const int encoding);
      token_t* getCharToken(const int encoding);

      int peekForHeader();
      bool loadingQuotedHeader();
      bool loadingAngleBracketHeader();
      std::string getHeader();

      void setOrigin(const int line,
                     const std::string &filename);

      static tokenVector tokenize(const std::string &source);

      static void tokenize(tokenVector &tokens,
                           fileOrigin origin,
                           const std::string &source);
    };
  }
}

#endif
