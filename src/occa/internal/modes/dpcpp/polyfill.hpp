#include <occa/defines.hpp>

#ifndef OCCA_MODES_DPCPP_POLYFILL_HEADER
#define OCCA_MODES_DPCPP_POLYFILL_HEADER

#if OCCA_DPCPP_ENABLED
#include <sycl.hpp>
#else
#include <vector>
namespace sycl {

class device;
class platform;

template <std::size_t N> 
struct id {
  std::size_t values_[N];
  std::size_t operator[](int dimension) const {return values_[dimension];}
};

template <std::size_t N>
struct range {
  std::size_t values_[N];
};

template <std::size_t N>
struct nd_range {
  range<N> global_range_;
  range<N> local_range_;
};

class exception : std::exception {
public:
  inline const char *what() const noexcept {
    return "Error-DPC++ not enabled!";
  }
};

namespace info {

enum class device_type {
  cpu,
  gpu,
  accelerator,
  all = (cpu | gpu | accelerator)
};

enum class event_profiling {
  command_submit,
  command_start,
  command_end
};

namespace device {

struct device_type {
  using return_type = info::device_type;
};

struct max_compute_units {
  using return_type = uint32_t;
};

struct global_mem_size {
  using return_type = uint64_t;
};

struct local_mem_size {
  using return_type = uint64_t;
};

struct name {
  using return_type = std::string;
};

struct vendor {
  using return_type = std::string;
};

struct version {
  using return_type = std::string;
};

template <int Dimensions = 3> struct max_work_item_sizes;
template<> struct max_work_item_sizes<3> {
  using return_type = id<3>;
};

struct max_work_group_size {
  using return_type = std::size_t;
};

} // namespace device

namespace platform {

struct name {
  using return_type = std::string;
};

struct vendor {
  using return_type = std::string;
};

struct version {
  using return_type = std::string;
};

} // namespace platform
} // namespace info


namespace property {
namespace queue {

struct enable_profiling {};
struct in_order {};

} // namespace queue
} // namespace property

class property_list {
public:
  template <typename... propertyTN> 
  property_list(propertyTN... props) {
    throw sycl::exception();
  }
};

class device {
public:
  static std::vector<device> get_devices() {
    throw sycl::exception();
    return std::vector<device>();
  }

  template <typename T>
  typename T::return_type get_info() const;
  
  bool is_cpu() const {
    throw sycl::exception();
    return false;
  }

  bool is_gpu() const {
    throw sycl::exception();
    return false;
  }

  bool is_accelerator() const {
    throw sycl::exception();
    return false;
  }
  
  sycl::platform get_platform() const;
};

template <>
inline info::device::max_compute_units::return_type
device::get_info<info::device::max_compute_units>() const {
  throw sycl::exception();
  return uint32_t(0);
}

template <>
inline info::device::global_mem_size::return_type
device::get_info<info::device::global_mem_size>() const {
  throw sycl::exception();
  return uint64_t(0);
}

template <>
inline info::device::local_mem_size::return_type
device::get_info<info::device::local_mem_size>() const {
  throw sycl::exception();
  return uint64_t(0);
}

template <>
inline info::device::name::return_type
device::get_info<info::device::name>() const {
  throw sycl::exception();
  return "Error--DPC++ not enabled!";
}

template <>
inline info::device::vendor::return_type
device::get_info<info::device::vendor>() const {
  throw sycl::exception();
  return "Error--DPC++ not enabled!";
}

template <>
inline info::device::version::return_type
device::get_info<info::device::version>() const {
  throw sycl::exception();
  return "Error--DPC++ not enabled!";
}

template <>
inline info::device::device_type::return_type
device::get_info<info::device::device_type>() const {
  throw sycl::exception();
  return info::device_type::all;
}

template <>
inline info::device::max_work_item_sizes<3>::return_type
device::get_info<info::device::max_work_item_sizes<3>>() const {
  throw sycl::exception();
  return sycl::id<3>{0,0,0};
}

template <>
inline info::device::max_work_group_size::return_type
device::get_info<info::device::max_work_group_size>() const {
  throw sycl::exception();
  return std::size_t(0);
}

class platform {
  public:
  static std::vector<platform> get_platforms() {
    throw sycl::exception();
    return std::vector<platform>();
  }

  std::vector<device> get_devices(info::device_type = info::device_type::all) {
    throw sycl::exception();
    return std::vector<device>();
  }

  template <typename T>
  typename T::return_type get_info() const;
};

template <>
inline info::platform::name::return_type
platform::get_info<info::platform::name>() const {
  throw sycl::exception();
  return "Error--DPC++ not enabled!";
}

template <>
inline info::platform::vendor::return_type
platform::get_info<info::platform::vendor>() const {
  throw sycl::exception();
  return "Error--DPC++ not enabled!";
}

template <>
inline info::platform::version::return_type
platform::get_info<info::platform::version>() const {
  throw sycl::exception();
  return "Error--DPC++ not enabled!";
}

inline platform device::get_platform() const {
    throw sycl::exception();
    return platform();
}

class context {
public:
  context() = default;
  context(const device &syclDevice){}
};

class event {
public:
  void wait_and_throw() {
    throw sycl::exception();
  }

  template <info::event_profiling I>
  uint64_t get_profiling_info() const {
    throw sycl::exception();
    return uint64_t(0);
  }
};

class queue {
public:
  queue(const context& syclContext,
        const device &syclDevice, 
        const property_list &propList = {}) {
    throw sycl::exception();
  }

  void wait_and_throw() {throw sycl::exception();}

  sycl::event memcpy(void* dest,const void* src,size_t num_bytes) {
    return sycl::event();
  }

  sycl::event ext_oneapi_submit_barrier()  { return sycl::event();}
};

inline void* malloc_device(size_t num_bytes, 
                    const device& syclDevice, 
                    const context& syclContext) {
  throw sycl::exception();
  return nullptr;
}

inline void* malloc_host(size_t num_bytes, 
                  const context& syclContext) {
  throw sycl::exception();
  return nullptr;
}

inline void* malloc_shared(size_t num_bytes, 
                    const device& syclDevice, 
                    const context& syclContext) {
  throw sycl::exception();
  return nullptr;
}

inline void free(void* ptr,context& syclContext) {
  throw sycl::exception();
}

} // namespace sycl
#endif

#endif
