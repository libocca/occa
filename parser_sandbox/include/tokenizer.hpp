/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 David Medina and Tim Warburton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */
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
    namespace encodingType {
      extern const int none;
      extern const int R;
      extern const int u8;
      extern const int u;
      extern const int U;
      extern const int L;
      extern const int ux;
    }

    namespace tokenType {
      extern const int none;

      extern const int identifier;

      extern const int systemHeader;
      extern const int header;

      extern const int primitive;
      extern const int op;

      extern const int withUType;
      extern const int withUDF;
      extern const int char_;
      extern const int string;
    }

    class token_t {
    public:
      fileOrigin origin;

      token_t(const fileOrigin &origin_);
      virtual ~token_t();

      virtual int type() const = 0;

      virtual void print(printer &pout) const = 0;
    };

    class identifierToken : public token_t {
    public:
      std::string value;

      identifierToken(const fileOrigin &origin_,
                      const std::string &value_);
      virtual ~identifierToken();

      virtual int type() const;

      virtual void print(printer &pout) const;
    };

    class primitiveToken : public token_t {
    public:
      primitive value;

      primitiveToken(const fileOrigin &origin_,
                     const primitive &value_);
      virtual ~primitiveToken();

      virtual int type() const;

      virtual void print(printer &pout) const;
    };

    class operatorToken : public token_t {
    public:
      const operator_t &op;

      operatorToken(const fileOrigin &origin_,
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

      charToken(const fileOrigin &origin_,
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

      stringToken(const fileOrigin &origin_,
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

      headerToken(const fileOrigin &origin_,
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

    class charStream {
    public:
      file_t *file;
      filePosition fp;
      std::vector<filePosition> stack;

      charStream(const char *root);
      charStream(file_t *file_,
                 const char *root);

      charStream(const charStream &stream);

      const char *getPosition();
      void setPosition(const char * pos);
      int getLine();
      void setLine(const int line);
      fileOrigin getFileOrigin();

      void push();
      void pop(const bool rewind = false);
      void popAndRewind();
      std::string str();

      void countSkippedLines(const char *start);

      void skipTo(const char delimiter);
      void skipTo(const char *delimiters);
      void skipFrom(const char *delimiters);

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
