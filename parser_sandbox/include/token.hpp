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
#if 0
#ifndef OCCA_PARSER_TOKEN_HEADER2
#define OCCA_PARSER_TOKEN_HEADER2

#include <iostream>

#include "occa/tools/gc.hpp"

/*
  Comments are replaced by a space ' '

  \\n -> nothing

  \n is guaranteed by the end of a file
  \s -> one space

  "a" "b" -> "ab"

  Make tokens
*/

class fileInfo {
public:
  std::string path;
  std::string source;

  fileInfo(const std::string &path_);
};

class fileInfoDB {
private:
  std::map<std::string, int> pathToID;
  std::map<int, fileInfo*> idToPath;
  int currentID;

public:
  fileInfoDB();
  fileInfoDB();
  ~fileInfoDB();

  const std::string& get(const std::string &path);
  const std::string& get(const int id);
};

namespace occa {
  namespace lang {
    class tokenStream;

    class token_t {
    public:
    };

    class tokenStream {
    private:
      char *start, *end;
      char *ptr;

    public:
      tokenStream();

      tokenStream(const char *start_,
                  const char *end_ = NULL);

      tokenStream(const std::string &str);

      tokenStream(const tokenStream &stream);

      virtual void destructor();

      void load(const char *start_,
                const char *end_ = NULL);

      void clear();

      bool hasNext();

      bool get(token_t &token);
    };
  }
}

#endif
#endif