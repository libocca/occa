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
#ifndef OCCA_PARSER_TOKENSTREAM_HEADER2
#define OCCA_PARSER_TOKENSTREAM_HEADER2

#include "occa/tools/io.hpp"

#include "file.hpp"
#include "printer.hpp"
#include "token.hpp"

namespace occa {
  namespace lang {
    class tokenStream;
    class tokenStreamTransform;

    //---[ tokenStream ]----------------
    class tokenStream : public errorHandler,
                        public withRefs {
      friend class tokenStreamWithMap;
      friend class tokenStreamTransform;
      friend class tokenStreamTransformWithMap;

    protected:
      tokenStream *sourceStream;

    public:
      virtual ~tokenStream();

      virtual token_t* getToken();

    protected:
      virtual token_t* _getToken();

      token_t* getSourceToken();
    };
    //==================================

    //---[ With Map ]-------------------
    class tokenStreamWithMap : virtual public tokenStream {
    private:
      std::vector<tokenStreamTransform*> transforms;

    public:
      virtual ~tokenStreamWithMap();

      void copyStreamTransforms(const tokenStreamWithMap &stream);

      virtual token_t* getToken();

      tokenStreamWithMap& map(tokenStreamTransform *transform);
    };
    //==================================

    //---[ Transform ]------------------
    class tokenStreamTransform : virtual public tokenStream {
    public:
      virtual ~tokenStreamTransform();
    };
    //==================================

    //---[ Transform With Map ]---------
    class tokenStreamTransformWithMap : virtual public tokenStream,
                                        public tokenStreamTransform,
                                        public tokenStreamWithMap {
      friend class tokenStreamWithMap;
    public:
      virtual ~tokenStreamTransformWithMap();
    };
    //==================================
  }
}

#endif
