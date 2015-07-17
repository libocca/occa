#include "occa/base.hpp"

namespace occa {
  // [--] Add uintptr_t support to occa::kernels

  template <class TM>
  void memset(void *ptr, const TM &value, uintptr_t count){
    OCCA_CHECK(false,
               "memset is only implemented for POD-type data (bool, char, short, int, long, float, double)");
  }

  template <>
  void memset<bool>(void *ptr, const bool &value, uintptr_t count);

  template <>
  void memset<char>(void *ptr, const char &value, uintptr_t count);

  template <>
  void memset<short>(void *ptr, const short &value, uintptr_t count);

  template <>
  void memset<int>(void *ptr, const int &value, uintptr_t count);

  template <>
  void memset<long>(void *ptr, const long &value, uintptr_t count);

  template <>
  void memset<float>(void *ptr, const float &value, uintptr_t count);

  template <>
  void memset<double>(void *ptr, const double &value, uintptr_t count);
}
