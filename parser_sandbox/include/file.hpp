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
#ifndef OCCA_PARSER_FILE_HEADER2
#define OCCA_PARSER_FILE_HEADER2

#include <iostream>
#include "occa/tools/gc.hpp"
#include "operator.hpp"

namespace occa {
  namespace lang {
    class file_t : public withRefs {
    public:
      std::string dirname;
      std::string filename;
      std::string content;

      file_t(const std::string &filename_);

      file_t(const std::string &dirname_,
             const std::string &filename_,
             const std::string &content_);
    };

    namespace originSource {
      extern file_t builtin;
      extern file_t string;
    }

    class filePosition {
    public:
      int line;
      const char *lineStart;
      const char *pos;

      filePosition();

      filePosition(const char *root);

      filePosition(const int line_,
                   const char *lineStart_,
                   const char *pos_);

      filePosition(const filePosition &other);
    };

    class fileOrigin : public withRefs,
                       public errorHandler {
    public:
      bool fromInclude;
      file_t *file;
      filePosition position;
      fileOrigin *up;

      fileOrigin();

      fileOrigin(file_t &file_);

      fileOrigin(const filePosition &position_);

      fileOrigin(file_t &file_,
                 const filePosition &position_);

      fileOrigin(const fileOrigin &other);

      fileOrigin& operator = (const fileOrigin &other);

      virtual ~fileOrigin();

      void setFile(file_t &file_);
      void setUp(fileOrigin *up_);

      void push(const bool fromInclude_,
                file_t &file_,
                const filePosition &position_);

      void pop();

      virtual void preprint(std::ostream &out);
      virtual void postprint(std::ostream &out);

      void print(std::ostream &out,
                 const bool root = true) const;

      inline void print(const bool root = true) const {
        print(std::cerr, root);
      }
    };
  }
}

#endif
