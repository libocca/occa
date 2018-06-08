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
#ifndef OCCA_LANG_SOURCETREAM_HEADER
#define OCCA_LANG_SOURCETREAM_HEADER

#include <list>
#include <vector>

#include <occa/io.hpp>
#include <occa/tools/trie.hpp>
#include <occa/lang/errorHandler.hpp>
#include <occa/lang/file.hpp>
#include <occa/lang/printer.hpp>
#include <occa/lang/stream.hpp>

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

    class tokenizer_t : public baseStream<token_t*>,
                        public errorHandler {
    public:
      tokenList outputCache;

      fileOrigin origin;
      filePosition &fp;

      originVector stack;

      operatorTrie operators;
      std::string operatorCharcodes;

      tokenizer_t();

      tokenizer_t(const char *root);

      tokenizer_t(file_t *file_);

      tokenizer_t(fileOrigin origin_);

      tokenizer_t(const tokenizer_t &stream);

      tokenizer_t& operator = (const tokenizer_t &stream);

      virtual ~tokenizer_t();

      void setup();

      virtual baseStream<token_t*>& clone() const;

      virtual void* passMessageToInput(const occa::properties &props);

      void set(const char *root);

      void set(file_t *file_);

      void clear();

      virtual void preprint(std::ostream &out);
      virtual void postprint(std::ostream &out);

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
      void getString(std::string &value,
                     const int encoding = 0);
      void getRawString(std::string &value);

      int skipLineCommentAndPeek();
      int skipBlockCommentAndPeek();

      token_t* getToken();
      token_t* getIdentifierToken();
      token_t* getPrimitiveToken();
      token_t* getOperatorToken();
      token_t* getStringToken(const int encoding);
      token_t* getCharToken(const int encoding);

      int peekForHeader();
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
