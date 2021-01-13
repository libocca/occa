#ifndef OCCA_INTERNAL_UTILS_TLS_HEADER
#define OCCA_INTERNAL_UTILS_TLS_HEADER

#include <occa/defines.hpp>

#include <iostream>

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
#  include <pthread.h>
#else
#  include <windows.h>
#  include <intrin.h>
#endif

namespace occa {
  template <class TM>
  class tls {
  private:
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
    pthread_key_t pkey;
#else
    thread_local TM value_;
#endif

  public:
    tls(const TM &val = TM());

    template <class TM2>
    tls(const tls<TM2> &t);

    ~tls();

    template <class TM2>
    const TM2& operator = (const TM2 &val);
    template <class TM2>
    const TM2& operator = (const tls<TM2> &t);

    TM& value();
    const TM& value() const;

    operator TM ();
    operator TM () const;
  };

  template <class TM>
  std::ostream& operator << (std::ostream &out,
                           const tls<TM> &t);
}

#include "tls.tpp"

#endif
