/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2017 David Medina and Tim Warburton
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

#ifndef OCCA_PAR_TLS_HEADER
#define OCCA_PAR_TLS_HEADER

#include "occa/defines.hpp"

#include <iostream>

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
#  include <pthread.h>
#else
#  include <windows.h>
#  include <intrin.h>
#endif

namespace occa {
  template <class TM>
  class tls {
  private:
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
    pthread_key_t pkey;
#else
    __declspec(thread) TM value_;
#endif

  public:
    tls(const TM &val = TM());
    template <class TM2>
    tls(const tls<TM2> &t);

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
  std::ostream& operator << (std::ostream &out, const tls<TM> &t);
}

#include "occa/par/tls.tpp"

#endif
