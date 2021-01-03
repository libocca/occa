namespace occa {
  template <class TM>
  occa::memory malloc(const dim_t entries,
                      const void *src,
                      const occa::json &props) {
    return malloc(entries, dtype::get<TM>(), src, props);
  }

  template <class TM>
  TM* umalloc(const dim_t entries,
              const void *src,
              const occa::json &props) {
    return (TM*) umalloc(entries, dtype::get<TM>(), src, props);
  }

  template <class TM>
  occa::memory wrapMemory(const TM *ptr,
                          const dim_t entries,
                          const occa::json &props) {
    return getDevice().wrapMemory(ptr, entries, occa::dtype::get<TM>(), props);
  }
}
