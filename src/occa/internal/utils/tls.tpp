namespace occa {
  template <class TM>
  tls<TM>::tls(const TM &val) {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
    pthread_key_create(&pkey, NULL);
    pthread_setspecific(pkey, new TM(val));
#else
    value_ = val;
#endif
  }

  template <class TM>
  template <class TM2>
  tls<TM>::tls(const tls<TM2> &t) {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
    pthread_key_create(&pkey, NULL);
    pthread_setspecific(pkey, new TM(t.value()));
#else
    value_ = t.value_;
#endif
  }

  template <class TM>
  tls<TM>::~tls() {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
    delete (TM*) pthread_getspecific(pkey);
    pthread_key_delete(pkey);
#endif
  }

  template <class TM>
  template <class TM2>
  const TM2& tls<TM>::operator = (const TM2 &val) {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
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
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
    delete &(value());
    pthread_setspecific(pkey, new TM(val));
#else
    value_ = val;
#endif
    return val;
  }

  template <class TM>
  TM& tls<TM>::value() {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
    return *((TM*) pthread_getspecific(pkey));
#else
    return value_;
#endif
  }

  template <class TM>
  const TM& tls<TM>::value() const {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
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
  std::ostream& operator << (std::ostream &out,
                           const tls<TM> &t) {
    out << t.value();
    return out;
  }
}
