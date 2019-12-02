namespace occa {
  template <class TM>
  occa::memory malloc(const dim_t entries,
                      const void *src,
                      const occa::properties &props) {
    return malloc(entries, dtype::get<TM>(), src, props);
  }

  template <class TM>
  TM* umalloc(const dim_t entries,
              const void *src,
              const occa::properties &props) {
    return (TM*) umalloc(entries, dtype::get<TM>(), src, props);
  }
}
