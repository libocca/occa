namespace occa {
namespace experimental {
  template <class T>
  occa::memory memoryPool::reserve(const dim_t entries) {
    return reserve(entries, occa::dtype::get<T>());
  }
}
}
