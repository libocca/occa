#include "occaBase.hpp"

namespace occa {
  // [--] Add uintptr_t support to occa::kernels

  template <class TM>
  void memset(void *ptr, const TM &value, uintptr_t count){
    OCCA_CHECK(false,
               "memset is only implemented for POD-type data (bool, char, short, int, long, float, double)");
  }

  void memsetBool(void *ptr, const bool &value, uintptr_t count);
  void memsetChar(void *ptr, const char &value, uintptr_t count);
  void memsetShort(void *ptr, const short &value, uintptr_t count);
  void memsetInt(void *ptr, const int &value, uintptr_t count);
  void memsetLong(void *ptr, const long &value, uintptr_t count);
  void memsetFloat(void *ptr, const float &value, uintptr_t count);
  void memsetDouble(void *ptr, const double &value, uintptr_t count);

  template <>
  inline void memset<bool>(void *ptr, const bool &value, uintptr_t count){
    memsetBool(ptr, value, count);
  }

  template <>
  inline void memset<char>(void *ptr, const char &value, uintptr_t count){
    memsetChar(ptr, value, count);
  }

  template <>
  inline void memset<short>(void *ptr, const short &value, uintptr_t count){
    memsetShort(ptr, value, count);
  }

  template <>
  inline void memset<int>(void *ptr, const int &value, uintptr_t count){
    memsetInt(ptr, value, count);
  }

  template <>
  inline void memset<long>(void *ptr, const long &value, uintptr_t count){
    memsetLong(ptr, value, count);
  }

  template <>
  inline void memset<float>(void *ptr, const float &value, uintptr_t count){
    memsetFloat(ptr, value, count);
  }

  template <>
  inline void memset<double>(void *ptr, const double &value, uintptr_t count){
    memsetDouble(ptr, value, count);
  }
};
