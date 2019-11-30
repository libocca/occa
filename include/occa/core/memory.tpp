namespace occa {
  template <class TM>
  TM* memory::ptr() {
    return (TM*) ptr<void>();
  }

  template <class TM>
  const TM* memory::ptr() const {
    return (const TM*) ptr<void>();
  }

  template <class TM>
  TM* memory::ptr(const occa::properties &props) {
    return (TM*) ptr<void>(props);
  }

  template <class TM>
  const TM* memory::ptr(const occa::properties &props) const {
    return (const TM*) ptr<void>(props);
  }
}
