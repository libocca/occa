namespace occa {
  template <class T>
  T* memory::ptr() {
    return (T*) ptr<void>();
  }

  template <class T>
  const T* memory::ptr() const {
    return (const T*) ptr<void>();
  }
}
