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

#ifndef OCCA_TYPES_BITFIELD_HEADER
#define OCCA_TYPES_BITFIELD_HEADER

#include <occa/types/typedefs.hpp>

namespace occa {
  class bitfield {
  public:
    udim_t b1, b2;

    inline bitfield() :
      b1(0),
      b2(0) {}

    inline bitfield(const udim_t b2_) :
      b1(0),
      b2(b2_) {}

    inline bitfield(const udim_t b1_,
                    const udim_t b2_) :
      b1(b1_),
      b2(b2_) {}

    inline static int bits() {
      // 2 * sizeof(udim_t) * 8
      return 16 * sizeof(udim_t);
    }

    inline bool operator == (const bitfield &bf) const {
      return ((b1 == bf.b1) &&
              (b2 == bf.b2));
    }

    inline bool operator != (const bitfield &bf) const {
      return ((b1 != bf.b1) ||
              (b2 != bf.b2));
    }

    inline bool operator < (const bitfield &bf) const {
      return ((b1 < bf.b1) ||
              ((b1 == bf.b1) &&
               (b2 < bf.b2)));
    }

    inline bool operator <= (const bitfield &bf) const {
      return ((b1 < bf.b1) ||
              ((b1 == bf.b1) &&
               (b2 <= bf.b2)));
    }

    inline bool operator > (const bitfield &bf) const {
      return ((b1 > bf.b1) ||
              ((b1 == bf.b1) &&
               (b2 > bf.b2)));
    }

    inline bool operator >= (const bitfield &bf) const {
      return ((b1 > bf.b1) ||
              ((b1 == bf.b1) &&
               (b2 >= bf.b2)));
    }

    inline bitfield operator | (const bitfield &bf) const {
      return bitfield(b1 | bf.b1,
                      b2 | bf.b2);
    }

    inline bitfield operator & (const bitfield &bf) const {
      return bitfield(b1 & bf.b1,
                      b2 & bf.b2);
    };

    inline bitfield operator << (const int shift) const {
      if (shift <= 0) {
        return *this;
      }
      const int bSize = 8 * sizeof(udim_t);
      if (shift > (2 * bSize)) {
        return 0;
      }

      if (shift >= bSize) {
        return bitfield(b2 << (shift - bSize),
                        0);
      }

      const udim_t carryOver = (b2 >> (bSize - shift));
      return bitfield((b1 << shift) | carryOver,
                      b2 << shift);
    };

    inline bitfield operator >> (const int shift) const {
      if (shift <= 0) {
        return *this;
      }
      const int bSize = 8 * sizeof(udim_t);
      if (shift > (2 * bSize)) {
        return 0;
      }

      if (shift >= bSize) {
        return bitfield(0,
                        b1 >> (shift - bSize));
      }

      const udim_t carryOver = (b1 << (bSize - shift));
      return bitfield(b1 >> shift,
                      (b2 >> shift) | carryOver);
    };

    inline bitfield& operator <<= (const int shift) {
      *this = (*this << shift);
      return *this;
    }

    inline bitfield& operator >>= (const int shift) {
      *this = (*this >> shift);
      return *this;
    }

    inline operator bool () const {
      return (b1 || b2);
    }
  };
}

#endif
