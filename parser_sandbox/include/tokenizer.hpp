#ifndef OCCA_PARSER_PREPROCESSOR_HEADER2
#define OCCA_PARSER_PREPROCESSOR_HEADER2

#include "occa/tools/io.hpp"

#include "trie.hpp"
#include "file.hpp"

namespace occa {
  namespace lang {
    class operator_t;

    typedef trie<const operator_t*> operatorTrie;

    namespace charcodes {
      extern const char whitespace[];
      extern const char alpha[];
      extern const char number[];
      extern const char alphanumber[];
      extern const char identifierStart[];
      extern const char identifier[];
      extern const char operators[];
    }

    operatorTrie& getOperators();

    //---[ Tokens ]---------------------
    class encodingType {
    public:
      static const int none = 0;
      static const int R    = (1 << 0);
      static const int u8   = (1 << 1);
      static const int u    = (1 << 2);
      static const int U    = (1 << 3);
      static const int L    = (1 << 4);
      static const int ux   = (u8 | u | U | L);
    };

    class tokenType {
    public:
      static const int none         = 0;

      static const int identifier   = (1 << 0);

      static const int systemHeader = (1 << 1);
      static const int header       = (3 << 1);

      static const int primitive    = (1 << 3);
      static const int op           = (1 << 4);

      static const int withUType    = (1 << 5);
      static const int withUDF      = (1 << 6);
      static const int char_        = (1 << 7);
      static const int string       = (1 << 8);
    };

    class token_t {
    public:
      fileOrigin *origin;

      token_t(fileOrigin *origin_);
      virtual ~token_t();

      virtual int type() const = 0;

      virtual void print(printer &pout) const = 0;
    };

    class identifierToken : public token_t {
    public:
      std::string value;

      identifierToken(fileOrigin *origin_,
                      const std::string &value_);
      virtual ~identifierToken();

      virtual int type() const;

      virtual void print(printer &pout) const;
    };

    class primitiveToken : public token_t {
    public:
      primitive value;

      primitiveToken(fileOrigin *origin_,
                     const primitive &value_);
      virtual ~primitiveToken();

      virtual int type() const;

      virtual void print(printer &pout) const;
    };

    class operatorToken : public token_t {
    public:
      const operator_t &op;

      operatorToken(fileOrigin *origin_,
                    const operator_t &op_);
      virtual ~operatorToken();

      virtual int type() const;

      virtual void print(printer &pout) const;
    };

    class charToken : public token_t {
    public:
      int uType;
      std::string value;
      std::string udf;

      charToken(fileOrigin *origin_,
                int uType_,
                const std::string &value_,
                const std::string &udf_);
      virtual ~charToken();

      virtual int type() const;

      virtual void print(printer &pout) const;
    };

    class stringToken : public token_t {
    public:
      int uType;
      std::string value;
      std::string udf;

      stringToken(fileOrigin *origin_,
                  int uType_,
                  const std::string &value_,
                  const std::string &udf_);
      virtual ~stringToken();

      virtual int type() const;

      virtual void print(printer &pout) const;
    };

    class headerToken : public token_t {
    public:
      bool systemHeader;
      std::string value;

      headerToken(fileOrigin *origin_,
                  const bool systemHeader_,
                  const std::string &value_);
      virtual ~headerToken();

      virtual int type() const;

      virtual void print(printer &pout) const;
    };
    //==================================

    //---[ Character Stream ]-----------
    int getEncodingType(const std::string &str);
    int getCharacterEncoding(const std::string &str);
    int getStringEncoding(const std::string &str);

    class charStringInfo {
    public:
      const char *root, *pos;
      int newlinesPassed;

      charStringInfo(const char *root_);
      charStringInfo(const charStringInfo &other);
    };

    class charStream {
    public:
      const char *root, *pos;
      int newlinesPassed;
      std::vector<charStringInfo> stack;

      charStream(const char *root_);

      charStream(const charStream &stream);

      const char *getPosition();
      void setPosition(const char * pos_);

      void clear();
      void push();
      void pop(const bool rewind = true);
      std::string str();

      void skipTo(const char delimiter);

      void skipTo(const char delimiter,
                  const char escape);

      void skipTo(const char *delimiters);

      void skipTo(const char *delimiters,
                  const char escape);

      void skipFrom(const char *delimiters);

      void skipFrom(const char *delimiters,
                    const char escape);

      void skipWhitespace();

      int peek();
      int shallowPeek();
      int peekForIdentifier();
      int peekForHeader();

      token_t* getToken();
      token_t* getIdentifierToken();
      token_t* getPrimitiveToken();
      token_t* getOperatorToken();
      token_t* getStringToken();
      token_t* getCharToken();
      token_t* getHeaderToken();
    };
    //==================================

    //---[ Tokenizer ]------------------
    class tokenizer_t {
    public:
      void parseFile(const std::string &filename) {
        char *c = io::c_read(filename);
        parse(filename, c);
      }

      void parseSource(const std::string &str) {
        parse("(source)", str.c_str());
      }

      void parse(const std::string &source,
                 const char *c) {
        while (*c) {

        }
      }
    };
    //==================================
  }
}

#endif
