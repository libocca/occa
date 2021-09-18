#ifndef OCCA_MODES_DPCPP_UTILS_HEADER
#define OCCA_MODES_DPCPP_UTILS_HEADER

#include <iostream>

#include <occa/internal/modes/dpcpp/polyfill.hpp>
#include <occa/internal/core/device.hpp>

namespace occa {

  namespace dpcpp {
    class device;
    class stream;
    class streamTag;

    constexpr int max_dimensions{3};
    constexpr int x_index{2};
    constexpr int y_index{1};
    constexpr int z_index{0};

    bool isEnabled();

    void setCompiler(occa::json &dpcpp_properties) noexcept;
    void setCompilerFlags(occa::json &dpcpp_properties) noexcept;
    void setSharedFlags(occa::json &dpcpp_properties) noexcept;
    void setLinkerFlags(occa::json &dpcpp_properties) noexcept;
    
    inline void setCompilerLinkerOptions(occa::json &dpcpp_properties) noexcept
    {
      setCompiler(dpcpp_properties);
      setCompilerFlags(dpcpp_properties);
      setSharedFlags(dpcpp_properties);
      setLinkerFlags(dpcpp_properties);
    }

    occa::dpcpp::device& getDpcppDevice(modeDevice_t* device_);
    occa::dpcpp::stream& getDpcppStream(const occa::stream& stream_);
    occa::dpcpp::streamTag &getDpcppStreamTag(const occa::streamTag& tag);

    occa::device wrapDevice(::sycl::device device,
                            const occa::properties &props = occa::properties());

    void warn(const ::sycl::exception &e,
               const std::string &filename,
               const std::string &function,
               const int line,
               const std::string &message);

    void error(const ::sycl::exception &e,
               const std::string &filename,
               const std::string &function,
               const int line,
               const std::string &message);
  }
}

#endif
