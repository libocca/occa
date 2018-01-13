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
#ifndef OCCA_PARSER_MACRO_HEADER2
#define OCCA_PARSER_MACRO_HEADER2

#include <vector>
#include <iostream>

namespace occa {
  class macroPart;
  typedef std::vector<macroPart> macroPartVector_t;

  namespace macroInfo {
    static const int string        = (1 << 0);
    static const int arg           = (1 << 1);
    static const int stringify     = (1 << 2);
    static const int concat        = (1 << 3);
    static const int variadic      = (1 << 4);

    static const int hasSpace      = (3 << 5);
    static const int hasLeftSpace  = (1 << 5);
    static const int hasRightSpace = (1 << 6);
  }

  //---[ Part ]-------------------------
  class macroPart {
  public:
    int info;
    std::string str;
    int argPos;

    macroPart(const int info_ = 0);
    macroPart(const char *c);
    macroPart(const std::string &str_);
  };
  //====================================

  //---[ Macro ]-------------------------
  class preprocessor_t;

  class macro_t {
  public:
    static const std::string VA_ARGS;

    const preprocessor_t *preprocessor;
    const char *macroStart;
    const char *localMacroStart;

    std::string name, source;

    int argc;
    mutable bool hasVarArgs;
    macroPartVector_t parts;

    int definedLine, undefinedLine;

    macro_t(const preprocessor_t *preprocessor_ = NULL);
    macro_t(const preprocessor_t *preprocessor_,
            char *c);
    macro_t(const preprocessor_t *preprocessor_,
            const char *c);

    void clear();

    void load(char *c);
    void load(const char *c);

    void loadName(char *&c);
    void loadArgs(char *&c,
                  macroPartVector_t &argNames,
                  const bool loadingArgNames = true) const;
    void setParts(char *&c,
                  macroPartVector_t &argNames);

    bool isFunctionLike() const;

    std::string expand(const char *c) const;
    virtual std::string expand(char *&c) const;

    std::string toString() const;
    operator std::string() const;
    void print() const;

    //---[ Messages ]-------------------
    void printError(const char *c,
                    const std::string &message) const;
    void printFatalError(const char *c,
                         const std::string &message) const;
    void printWarning(const char *c,
                      const std::string &message) const;
    //==================================
  };
  //====================================
}

#endif
