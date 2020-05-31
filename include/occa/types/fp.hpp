#ifndef OCCA_TYPES_FP_HEADER
#define OCCA_TYPES_FP_HEADER

namespace occa {
  template<typename T1, typename T2>
  bool areBitwiseEqual(T1 a, T2 b) {
    if (sizeof(T1) != sizeof(T2)) return false;
    unsigned char *a_int = reinterpret_cast<unsigned char *>(&a);
    unsigned char *b_int = reinterpret_cast<unsigned char *>(&b);
    for (size_t i = 0; i < sizeof(T1); ++i) {
      if ((a_int[i] ^ b_int[i]) != 0) return false;
    }
    return true;
  }
}
#endif
