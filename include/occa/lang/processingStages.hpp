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
#ifndef OCCA_LANG_PROCESSINGSTAGES_HEADER
#define OCCA_LANG_PROCESSINGSTAGES_HEADER

#include <occa/lang/stream.hpp>

namespace occa {
  namespace lang {
    class token_t;

    typedef streamFilter<token_t*>              tokenFilter;
    typedef streamMap<token_t*, token_t*>       tokenMap;
    typedef withInputCache<token_t*, token_t*>  tokenInputCacheMap;
    typedef withOutputCache<token_t*, token_t*> tokenOutputCacheMap;

    class newlineTokenFilter : public tokenFilter {
    public:
      newlineTokenFilter();

      virtual tokenMap& clone_() const;
      virtual bool isValid(token_t * const &token);
    };

    class stringTokenMerger : public tokenOutputCacheMap {
    public:
      stringTokenMerger();
      stringTokenMerger(const stringTokenMerger &other);

      virtual tokenMap& clone_() const;
      virtual void fetchNext();
    };

    class externTokenMerger : public tokenInputCacheMap,
                              public tokenOutputCacheMap {
    public:
      externTokenMerger();
      externTokenMerger(const externTokenMerger &other);

      virtual tokenMap& clone_() const;
      virtual void fetchNext();
    };

    class unknownTokenFilter : public tokenFilter {
    public:
      bool printError;

      unknownTokenFilter(const bool printError_);

      tokenMap& clone_() const;

      void setPrintError(const bool printError_);

      virtual bool isValid(token_t * const &token);
    };
  }
}

#endif
