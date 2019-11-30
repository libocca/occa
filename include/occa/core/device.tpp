namespace occa {
  template <class TM>
  occa::memory device::malloc(const dim_t entries,
                              const void *src,
                              const occa::properties &props) {
    return malloc(entries, occa::dtype::get<TM>(), src, props);
  }

  template <class TM>
  occa::memory device::malloc(const dim_t entries,
                              const occa::memory src,
                              const occa::properties &props) {
    return malloc(entries, occa::dtype::get<TM>(), src, props);
  }

  template <class TM>
  occa::memory device::malloc(const dim_t entries,
                              const occa::properties &props) {
    return malloc(entries, occa::dtype::get<TM>(), props);
  }

  template <class TM>
  TM* device::umalloc(const dim_t entries,
                      const void *src,
                      const occa::properties &props) {
    return (TM*) umalloc(entries, dtype::get<TM>(), src, props);
  }

  template <class TM>
  TM* device::umalloc(const dim_t entries,
                      const occa::memory src,
                      const occa::properties &props) {
    return (TM*) umalloc(entries, dtype::get<TM>(), src, props);
  }

  template <class TM>
  TM* device::umalloc(const dim_t entries,
                      const occa::properties &props) {
    return (TM*) umalloc(entries, dtype::get<TM>(), props);
  }
}
