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
#ifndef OCCA_LANG_SPECIALMACROS_HEADER
#define OCCA_LANG_SPECIALMACROS_HEADER

#include <occa/lang/macro.hpp>

namespace occa {
  namespace lang {
    // defined()
    class definedMacro : public macro_t {
    public:
      definedMacro(preprocessor_t &pp_);

      virtual macro_t& clone(preprocessor_t &pp_) const;

      virtual void expand(tokenVector &tokens,
                          identifierToken &source);
    };

    // __has_include()
    class hasIncludeMacro : public macro_t {
    public:
      hasIncludeMacro(preprocessor_t &pp_);

      virtual macro_t& clone(preprocessor_t &pp_) const;

      virtual void expand(tokenVector &tokens,
                          identifierToken &source);
    };

    // __FILE__
    class fileMacro : public macro_t {
    public:
      fileMacro(preprocessor_t &pp_);

      virtual macro_t& clone(preprocessor_t &pp_) const;

      virtual void expand(tokenVector &tokens,
                          identifierToken &source);
    };

    // __LINE__
    class lineMacro : public macro_t {
    public:
      lineMacro(preprocessor_t &pp_);

      virtual macro_t& clone(preprocessor_t &pp_) const;

      virtual void expand(tokenVector &tokens,
                          identifierToken &source);
    };

    // __DATE__
    class dateMacro : public macro_t {
    public:
      dateMacro(preprocessor_t &pp_);

      virtual macro_t& clone(preprocessor_t &pp_) const;

      virtual void expand(tokenVector &tokens,
                          identifierToken &source);
    };

    // __TIME__
    class timeMacro : public macro_t {
    public:
      timeMacro(preprocessor_t &pp_);

      virtual macro_t& clone(preprocessor_t &pp_) const;

      virtual void expand(tokenVector &tokens,
                          identifierToken &source);
    };

    // __COUNTER__
    class counterMacro : public macro_t {
    public:
      mutable int counter;

      counterMacro(preprocessor_t &pp_,
                   const int counter_ = 0);

      virtual macro_t& clone(preprocessor_t &pp_) const;

      virtual void expand(tokenVector &tokens,
                          identifierToken &source);
    };
  }
}

#endif
