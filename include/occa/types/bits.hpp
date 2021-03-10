#ifndef OCCA_TYPES_BITS_HEADER
#define OCCA_TYPES_BITS_HEADER

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

  template<typename T1, typename T2>
  bool areBitwiseEqual(T1 a, T2 b) {
    if (sizeof(T1) != sizeof(T2)) {
      return false;
    }

    unsigned char *a_int = reinterpret_cast<unsigned char*>(&a);
    unsigned char *b_int = reinterpret_cast<unsigned char*>(&b);

    for (size_t i = 0; i < sizeof(T1); ++i) {
      if (a_int[i] != b_int[i]) {
        return false;
      }
    }

    return true;
  }
}

#endif
