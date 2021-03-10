namespace occa {
  template <class T>
  occa::memory device::malloc(const dim_t entries,
                              const void *src,
                              const occa::json &props) {
    return malloc(entries, occa::dtype::get<T>(), src, props);
  }

  template <class T>
  occa::memory device::malloc(const dim_t entries,
                              const occa::memory src,
                              const occa::json &props) {
    return malloc(entries, occa::dtype::get<T>(), src, props);
  }

  template <class T>
  occa::memory device::malloc(const dim_t entries,
                              const occa::json &props) {
    return malloc(entries, occa::dtype::get<T>(), props);
  }

  template <class T>
  T* device::umalloc(const dim_t entries,
                      const void *src,
                      const occa::json &props) {
    return (T*) umalloc(entries, dtype::get<T>(), src, props);
  }

  template <class T>
  T* device::umalloc(const dim_t entries,
                      const occa::memory src,
                      const occa::json &props) {
    return (T*) umalloc(entries, dtype::get<T>(), src, props);
  }

  template <class T>
  T* device::umalloc(const dim_t entries,
                      const occa::json &props) {
    return (T*) umalloc(entries, dtype::get<T>(), props);
  }

  template <>
  occa::memory device::wrapMemory<void>(const void *ptr,
                                        const dim_t entries,
                                        const occa::json &props);

  template <class T>
  occa::memory device::wrapMemory(const T *ptr,
                                  const dim_t entries,
                                  const occa::json &props) {
    return wrapMemory(ptr, entries, occa::dtype::get<T>(), props);
  }
}
