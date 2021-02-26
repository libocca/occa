#ifndef OCCA_MODES_DPCPP_UTILS_HEADER
#define OCCA_MODES_DPCPP_UTILS_HEADER

#include <iostream>

#include <occa/internal/modes/dpcpp/polyfill.hpp>
#include <occa/internal/core/device.hpp>
#include <occa/internal/io/lock.hpp>

namespace occa {

  namespace dpcpp {
    class device;
    class stream;
    class streamTag;

    constexpr int max_dimensions{3};
    constexpr int x_index{2};
    constexpr int y_index{1};
    constexpr int z_index{0};

    namespace info {
      static const int CPU     = (1 << 0);
      static const int GPU     = (1 << 1);
      static const int FPGA    = (1 << 2);
      //static const int XeonPhi = (1 << 2);
      static const int anyType = (CPU | GPU | FPGA);

      static const int Intel     = (1 << 3);
      static const int AMD       = (1 << 4);
      static const int Altera    = (1 << 5);
      static const int NVIDIA    = (1 << 6);
      static const int anyVendor = (Intel | AMD | Altera | NVIDIA);

      static const int any = (anyType | anyVendor);

      std::string deviceType(int type);
      std::string vendor(int type);
    }

    bool isEnabled();

    ::sycl::info::device_type deviceType(int type);

    int getPlatformID(const occa::json &properties);
    int getDeviceID(const occa::json &properties);

    int getPlatformCount();
    ::sycl::platform getPlatformByID(int pID);

    int getDeviceCount(int type = info::anyType);
    int getDeviceCountInPlatform(int pID, int type = info::anyType);
    ::sycl::device getDeviceByID(int pID, int dID, int type = info::anyType);

    std::string deviceName(int pID, int dID);

    int deviceType(int pID, int dID);
    int deviceVendor(int pID, int dID);
    int deviceCoreCount(int pID, int dID);

    udim_t getDeviceMemorySize(const ::sycl::device &devPtr);
    udim_t getDeviceMemorySize(int pID, int dID);

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
