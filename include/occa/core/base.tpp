namespace occa {
  template <class T>
  occa::memory malloc(const dim_t entries,
                      const void *src,
                      const occa::json &props) {
    return malloc(entries, dtype::get<T>(), src, props);
  }

  template <class T>
  T* umalloc(const dim_t entries,
              const void *src,
              const occa::json &props) {
    return (T*) umalloc(entries, dtype::get<T>(), src, props);
  }

  template <class T>
  occa::memory wrapMemory(const T *ptr,
                          const dim_t entries,
                          const occa::json &props) {
    return getDevice().wrapMemory(ptr, entries, occa::dtype::get<T>(), props);
  }
}
