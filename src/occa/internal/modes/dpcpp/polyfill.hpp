#include <occa/defines.hpp>

#ifndef OCCA_MODES_DPCPP_POLYFILL_HEADER
#define OCCA_MODES_DPCPP_POLYFILL_HEADER

#if OCCA_DPCPP_ENABLED
#include <sycl.hpp>
#else
#include <vector>
namespace sycl
{
  class device;
  class platform;

  template <std::size_t N>
  struct id
  {
    std::size_t values_[N];

    std::size_t operator[](int dimension) const
    {
      return values_[dimension];
    }
  };

  template <std::size_t N>
  struct range
  {
    std::size_t values_[N];
  };

  template <std::size_t N>
  struct nd_range
  {
    range<N> global_range_;
    range<N> local_range_;
  };

  class exception : std::exception
  {
    public:
    inline const char *what() const noexcept
    {
      return "Error-DPC++ not enabled!";
    }
  };

  namespace info
  {
    enum class device
    {
      device_type,
      max_compute_units,
      global_mem_size,
      local_mem_size,
      platform,
      name,
      vendor,
      version,
      max_work_item_sizes,
      max_work_group_size
    };

    enum class device_type
    {
      cpu,
      gpu,
      accelerator,
      all = (cpu | gpu | accelerator)
    };

    enum class platform
    {
      profile,
      version,
      name,
      vendor
    };

    enum class event_profiling
    {
      command_submit,
      command_start,
      command_end
    };

    template <typename T, T param>
    class param_traits
    {
    };

    template <>
    class param_traits<device, device::max_compute_units>
    {
    public:
      using return_type = uint32_t;
    };

    template <>
    class param_traits<device, device::global_mem_size>
    {
    public:
      using return_type = uint64_t;
    };

    template <>
    class param_traits<device, device::local_mem_size>
    {
    public:
      using return_type = uint64_t;
    };

    template <>
    class param_traits<device, device::name>
    {
    public:
      using return_type = std::string;
    };

    template <>
    class param_traits<device, device::vendor>
    {
    public:
      using return_type = std::string;
    };

    template <>
    class param_traits<device, device::version>
    {
    public:
      using return_type = std::string;
    };

    template <>
    class param_traits<device, device::max_work_item_sizes>
    {
    public:
      using return_type = id<3>;
    };

    template <>
    class param_traits<device, device::max_work_group_size>
    {
    public:
      using return_type = std::size_t;
    };

    template <>
    class param_traits<device, device::device_type>
    {
    public:
      using return_type = info::device_type;
    };

    template <>
    class param_traits<platform, platform::name>
    {
    public:
      using return_type = std::string;
    };

    template <>
    class param_traits<platform, platform::vendor>
    {
    public:
      using return_type = std::string;
    };

    template <>
    class param_traits<platform, platform::version>
    {
    public:
      using return_type = std::string;
    };

    template <event_profiling I>
    class param_traits<event_profiling, I>
    {
    public:
      using return_type = uint64_t;
    };
  } // namespace info

  namespace property {
    namespace queue {
      class enable_profiling
      {
      };
    }
  }

  namespace property {
    namespace queue {
      class in_order
      {
      };
    }
  }

  class property_list {
    public:
    template <typename... propertyTN> property_list(propertyTN... props)
    {
      throw sycl::exception();
    }
  };

  class device
  {
  public:
    static std::vector<device> get_devices()
    {
      throw sycl::exception();
      return std::vector<device>();
    }

    template <info::device I>
    typename info::param_traits<info::device, I>::return_type get_info() const;

    bool is_cpu() const
    {
      throw sycl::exception();
      return false;
    }

    bool is_gpu() const
    {
      throw sycl::exception();
      return false;
    }

    bool is_accelerator() const
    {
      throw sycl::exception();
      return false;
    }

    bool is_host() const
    {
      throw sycl::exception();
      return false;  
    }

    sycl::platform get_platform() const;
  };

  template <>
  inline info::param_traits<info::device, info::device::max_compute_units>::return_type
  device::get_info<info::device::max_compute_units>() const
  {
    throw sycl::exception();
    return uint32_t(0);
  }

  template <>
  inline info::param_traits<info::device, info::device::global_mem_size>::return_type
  device::get_info<info::device::global_mem_size>() const
  {
    throw sycl::exception();
    return uint64_t(0);
  }

  template <>
  inline info::param_traits<info::device, info::device::local_mem_size>::return_type
  device::get_info<info::device::local_mem_size>() const
  {
    throw sycl::exception();
    return uint64_t(0);
  }

  template <>
  inline info::param_traits<info::device, info::device::name>::return_type
  device::get_info<info::device::name>() const
  {
    throw sycl::exception();
    return "Error--DPC++ not enabled!";
  }

  template <>
  inline info::param_traits<info::device, info::device::vendor>::return_type
  device::get_info<info::device::vendor>() const
  {
    throw sycl::exception();
    return "Error--DPC++ not enabled!";
  }

  template <>
  inline info::param_traits<info::device, info::device::version>::return_type
  device::get_info<info::device::version>() const
  {
    throw sycl::exception();
    return "Error--DPC++ not enabled!";
  }

  template <>
  inline info::param_traits<info::device, info::device::device_type>::return_type
  device::get_info<info::device::device_type>() const
  {
    throw sycl::exception();
    return info::device_type::all;
  }

  template <>
  inline info::param_traits<info::device, info::device::max_work_item_sizes>::return_type
  device::get_info<info::device::max_work_item_sizes>() const
  {
    throw sycl::exception();
    return sycl::id<3>{0,0,0};
  }

  template <>
  inline info::param_traits<info::device, info::device::max_work_group_size>::return_type
  device::get_info<info::device::max_work_group_size>() const
  {
    throw sycl::exception();
    return std::size_t(0);
  }

  class platform
  {
    public:
    static std::vector<platform> get_platforms()
    {
      throw sycl::exception();
      return std::vector<platform>();
    }

    std::vector<device> get_devices(info::device_type = info::device_type::all)
    {
      throw sycl::exception();
      return std::vector<device>();
    }

    template <info::platform I>
    typename info::param_traits<info::platform, I>::return_type get_info() const;

    bool is_host() const
    {
      throw sycl::exception();
      return false;  
    }
  };

  template <>
  inline info::param_traits<info::platform, info::platform::name>::return_type
  platform::get_info<info::platform::name>() const
  {
    throw sycl::exception();
    return "Error--DPC++ not enabled!";
  }

  template <>
  inline info::param_traits<info::platform, info::platform::vendor>::return_type
  platform::get_info<info::platform::vendor>() const
  {
    throw sycl::exception();
    return "Error--DPC++ not enabled!";
  }

  template <>
  inline info::param_traits<info::platform, info::platform::version>::return_type
  platform::get_info<info::platform::version>() const
  {
    throw sycl::exception();
    return "Error--DPC++ not enabled!";
  }

  inline platform device::get_platform() const
  {
      throw sycl::exception();
      return platform();
  }

  class context
  {
    public:
      context() = default;
      context(const device &syclDevice)
      {
    }
  };

  class event
  {
    public:
      void wait_and_throw() 
      {
        throw sycl::exception();
      }

    template <info::event_profiling I>
    typename info::param_traits<info::event_profiling, I>::return_type get_profiling_info() const
    {
      throw sycl::exception();
      return uint64_t(0);
    }
  };

  // template <info::event_profiling I>
  // inline info::param_traits<info::event_profiling, I>::return_type
  // event::get_profiling_info() const
  // {
  //   throw sycl::exception();
  //   return "Error--DPC++ not enabled!";
  // }
 

  class queue
  {
    public:
      queue(const context& syclContext,
            const device &syclDevice, 
            const property_list &propList = {})
      {
        throw sycl::exception();
      }

      void wait_and_throw()
      {
        throw sycl::exception();
      }

      sycl::event memcpy(void* dest,const void* src,size_t num_bytes)
      {
        return sycl::event();
      }

      sycl::event ext_oneapi_submit_barrier() 
      {
        return sycl::event();
      }
  };


  inline void* malloc_device(size_t num_bytes, 
                      const device& syclDevice, 
                      const context& syclContext)
  {
    throw sycl::exception();
    return nullptr;
  }

  inline void* malloc_host(size_t num_bytes, 
                    const context& syclContext)
  {
    throw sycl::exception();
    return nullptr;
  }

  inline void* malloc_shared(size_t num_bytes, 
                      const device& syclDevice, 
                      const context& syclContext)
  {
    throw sycl::exception();
    return nullptr;
  }

  inline void free(void* ptr,context& syclContext)
  {
    throw sycl::exception();
  }

} // namespace sycl
#endif

#endif
