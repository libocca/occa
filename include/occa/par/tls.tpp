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

#include "occa/par/tls.hpp"

namespace occa {
  template <class TM>
  tls<TM>::tls(const TM &val) {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
    pthread_key_create(&pkey, NULL);
    pthread_setspecific(pkey, new TM(val));
#else
    value_ = val;
#endif
  }

  template <class TM>
  template <class TM2>
  tls<TM>::tls(const tls<TM2> &t) {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
    pthread_key_create(&pkey, NULL);
    pthread_setspecific(pkey, new TM(t.value()));
#else
    value_ = t.value_;
#endif
  }

  template <class TM>
  template <class TM2>
  const TM2& tls<TM>::operator = (const TM2 &val) {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
    delete &(value());
    pthread_setspecific(pkey, new TM(val));
#else
    value_ = val;
#endif
    return val;
  }

  template <class TM>
  template <class TM2>
  const TM2& tls<TM>::operator = (const tls<TM2> &t) {
    const TM2 &val = t.value();
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
    delete &(value());
    pthread_setspecific(pkey, new TM(val));
#else
    value_ = val;
#endif
    return val;
  }

  template <class TM>
  TM& tls<TM>::value() {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
    return *((TM*) pthread_getspecific(pkey));
#else
    return value_;
#endif
  }

  template <class TM>
  const TM& tls<TM>::value() const {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
    return *((TM*) pthread_getspecific(pkey));
#else
    return value_;
#endif
  }

  template <class TM>
  tls<TM>::operator TM () {
    return value();
  }

  template <class TM>
  tls<TM>::operator TM () const {
    return value();
  }

  template <class TM>
  std::ostream& operator << (std::ostream &out, const tls<TM> &t) {
    out << t.value();
    return out;
  }
}
