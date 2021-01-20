namespace occa {
  template <class TM>
  TM* memory::ptr() {
    return (TM*) ptr<void>();
  }

  template <class TM>
  const TM* memory::ptr() const {
    return (const TM*) ptr<void>();
  }
}
