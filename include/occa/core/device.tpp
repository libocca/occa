namespace occa {
  template <class TM>
  occa::memory device::malloc(const dim_t entries,
                              const void *src,
                              const occa::json &props) {
    return malloc(entries, occa::dtype::get<TM>(), src, props);
  }

  template <class TM>
  occa::memory device::malloc(const dim_t entries,
                              const occa::memory src,
                              const occa::json &props) {
    return malloc(entries, occa::dtype::get<TM>(), src, props);
  }

  template <class TM>
  occa::memory device::malloc(const dim_t entries,
                              const occa::json &props) {
    return malloc(entries, occa::dtype::get<TM>(), props);
  }

  template <class TM>
  TM* device::umalloc(const dim_t entries,
                      const void *src,
                      const occa::json &props) {
    return (TM*) umalloc(entries, dtype::get<TM>(), src, props);
  }

  template <class TM>
  TM* device::umalloc(const dim_t entries,
                      const occa::memory src,
                      const occa::json &props) {
    return (TM*) umalloc(entries, dtype::get<TM>(), src, props);
  }

  template <class TM>
  TM* device::umalloc(const dim_t entries,
                      const occa::json &props) {
    return (TM*) umalloc(entries, dtype::get<TM>(), props);
  }

  template <>
  occa::memory device::wrapMemory<void>(const void *ptr,
                                        const dim_t entries,
                                        const occa::json &props);

  template <class TM>
  occa::memory device::wrapMemory(const TM *ptr,
                                  const dim_t entries,
                                  const occa::json &props) {
    return wrapMemory(ptr, entries, occa::dtype::get<TM>(), props);
  }
}
