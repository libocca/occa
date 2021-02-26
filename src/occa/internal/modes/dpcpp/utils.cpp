#include <stdio.h>

#include <occa/internal/modes/dpcpp/utils.hpp>
#include <occa/internal/modes/dpcpp/device.hpp>
#include <occa/internal/modes/dpcpp/memory.hpp>
#include <occa/internal/modes/dpcpp/kernel.hpp>
#include <occa/internal/modes/dpcpp/stream.hpp>
#include <occa/internal/modes/dpcpp/streamTag.hpp>
#include <occa/internal/io.hpp>
#include <occa/internal/utils/sys.hpp>
#include <occa/internal/utils/env.hpp>
#include <occa/core/base.hpp>

namespace occa
{
  namespace dpcpp
  {
    namespace info
    {
      std::string deviceType(int type)
      {
        if (type & anyType)
          return "ALL";
        if (type & CPU)
          return "CPU";
        if (type & GPU)
          return "GPU";
        if (type & FPGA)
          return "FPGA";
        //if (type & XeonPhi) return "Xeon Phi";

        return "N/A";
      }

      std::string vendor(int type)
      {
        if (type & Intel)
          return "Intel";
        if (type & AMD)
          return "AMD";
        if (type & NVIDIA)
          return "NVIDIA";
        if (type & Altera)
          return "Altera";

        return "N/A";
      }
    } // namespace info

    /* Returns true if any DPC++ device is enabled on the machine */
    bool isEnabled()
    {
      bool isenabled = false;
      auto platformlist = ::sycl::platform::get_platforms();
      for (auto p : platformlist)
      {
        auto devicelist = p.get_devices(::sycl::info::device_type::all);
        if (devicelist.size() > 0)
          isenabled = true;
      }
      return isenabled;
    }

    /* Returns the DPC++ device type*/
    ::sycl::info::device_type deviceType(int type)
    {

      if (type & info::anyType)
        return ::sycl::info::device_type::all;
      if (type & info::CPU)
        return ::sycl::info::device_type::cpu;
      if (type & info::GPU)
        return ::sycl::info::device_type::gpu;
      if (type & info::FPGA)
        return ::sycl::info::device_type::accelerator;

      return ::sycl::info::device_type::all;
    }

    int getPlatformID(const occa::json &properties)
    {
      OCCA_ERROR("[DPCPP] No integer [platform_id] given",
                 properties.has("platform_id") &&
                     properties["platform_id"].isNumber());

      return properties.get<int>("platform_id");
    }

    int getDeviceID(const occa::json &properties)
    {
      OCCA_ERROR("[DPCPP] No integer [device_id] given",
                 properties.has("device_id") &&
                     properties["device_id"].isNumber());

      return properties.get<int>("device_id");
    }

    /* Returns the number of DPC++ platforms*/
    int getPlatformCount()
    {
      return ::sycl::platform::get_platforms().size();
    }

    /* Returns the DPC++ platform of interest */
    ::sycl::platform getPlatformByID(int pID)
    {
      auto platforms{::sycl::platform::get_platforms()};
      OCCA_ERROR("Invalid platform number (" + occa::toString(pID) + ")",
            (static_cast<size_t>(pID) < platforms.size()));
      return platforms[pID];
    }

    /* Returns the number of DPC++ devices of a certain device type*/
    int getDeviceCount(int type)
    {
      auto platformlist = ::sycl::platform::get_platforms();
      int count = 0;
      for (auto p : platformlist)
      {
        count += p.get_devices(deviceType(type)).size();
      }
      return count;
    }

    /* Return the number of DPC++ devices under a given platform */
    int getDeviceCountInPlatform(int pID, int type)
    {
      auto p = getPlatformByID(pID);
      
      return p.get_devices(deviceType(type)).size();
    }

    /* Return the DPC++ device given the platform ID and Device ID */
    ::sycl::device getDeviceByID(int pID, int dID, int type)
    {
      auto p = getPlatformByID(pID);
      auto device_type = deviceType(type);
      auto devices{p.get_devices(device_type)};
      OCCA_ERROR("Invalid device number (" + occa::toString(dID) + ")",
                 (static_cast<size_t>(dID) < devices.size()));
      return (devices[dID]);
    }
    /* Return the DPC++ device name */
    std::string deviceName(int pID, int dID)
    {
      auto d = getDeviceByID(pID, dID);
      return d.get_info<::sycl::info::device::name>();
    }
    /* Return the DPC++ device type */
    int deviceType(int pID, int dID)
    {
      auto d = getDeviceByID(pID, dID);
      return static_cast<int>(d.get_info<::sycl::info::device::device_type>());
    }

    /* Return the DPC++ device vendor */
    int deviceVendor(int pID, int dID)
    {
      auto d = getDeviceByID(pID, dID);
      std::string devVendor{d.get_info<::sycl::info::device::vendor>()};
      if (devVendor.find("Intel") != std::string::npos)
        return info::Intel;
      else if (devVendor.find("NVIDIA") != std::string::npos)
        return info::NVIDIA;
      else if (devVendor.find("Altera") != std::string::npos)
        return info::Altera;
      else if (devVendor.find("AMD") != std::string::npos)
        return info::AMD;
      return 0; //TODO check this
    }
    /* Returns the DPC++ Core count */
    int deviceCoreCount(int pID, int dID)
    {
      auto d = getDeviceByID(pID, dID);
      return d.get_info<::sycl::info::device::max_compute_units>();
    }

    /* Returns the DPC++ global memory size given the DPC++ device */
    udim_t getDeviceMemorySize(const ::sycl::device &devPtr)
    {
      return devPtr.get_info<::sycl::info::device::global_mem_size>();
    }
    /* Returns the DPC++ global memory size given the platform and device IDs */
    udim_t getDeviceMemorySize(int pID, int dID)
    {
      auto d = getDeviceByID(pID, dID);
      return getDeviceMemorySize(d);
    }

void setCompiler(occa::json &dpcpp_properties) noexcept
    {
      std::string compiler;
      if (env::var("OCCA_DPCPP_COMPILER").size())
      {
        compiler = env::var("OCCA_DPCPP_COMPILER");
      }
      else if (dpcpp_properties.has("compiler"))
      {
        compiler = dpcpp_properties["compiler"].toString();
      }
      else
      {
        compiler = "dpcpp";
      }
      dpcpp_properties["compiler"] = compiler;
    }

    void setCompilerFlags(occa::json &dpcpp_properties) noexcept
    {
      std::string compiler_flags;
      if (env::var("OCCA_DPCPP_COMPILER_FLAGS").size())
      {
        compiler_flags = env::var("OCCA_DPCPP_COMPILER_FLAGS");
      }
      else if (dpcpp_properties.has("compiler_flags"))
      {
        compiler_flags = dpcpp_properties["compiler_flags"].toString();
      }
      dpcpp_properties["compiler_flags"] = compiler_flags;
    }

void setSharedFlags(occa::json &dpcpp_properties) noexcept
    {
      std::string shared_flags;
      if (env::var("OCCA_COMPILER_SHARED_FLAGS").size())
      {
        shared_flags = env::var("OCCA_COMPILER_SHARED_FLAGS");
      }
      else if (dpcpp_properties.has("compiler_shared_flags"))
      {
        shared_flags = (std::string) dpcpp_properties["compiler_shared_flags"];
      }
      else
      {
        shared_flags = "-shared -fPIC";
      }
      dpcpp_properties["compiler_shared_flags"] = shared_flags;
    }

    void setLinkerFlags(occa::json &dpcpp_properties) noexcept
    {
      std::string linker_flags;
      if (env::var("OCCA_DPCPP_LINKER_FLAGS").size())
      {
        linker_flags = env::var("OCCA_DPCPP_LINKER_FLAGS");
      }
      else if (dpcpp_properties.has("linker_flags"))
      {
        linker_flags = dpcpp_properties["linker_flags"].toString();
      }
      dpcpp_properties["linker_flags"] = linker_flags;
    }

    occa::dpcpp::device& getDpcppDevice(modeDevice_t* device_)
    {
      occa::dpcpp::device* dpcppDevice = dynamic_cast<occa::dpcpp::device*>(device_);
      OCCA_ERROR("[dpcpp::getDpcppDevice] Dynamic cast failed!",nullptr != dpcppDevice);
      return *dpcppDevice;
    }  

    occa::dpcpp::stream& getDpcppStream(const occa::stream& stream_)
    {
      auto* dpcpp_stream{dynamic_cast<occa::dpcpp::stream*>(stream_.getModeStream())};
      OCCA_ERROR("[dpcpp::getDpcppStream]: Dynamic cast failed!", nullptr != dpcpp_stream);
      return *dpcpp_stream;
    }

    occa::dpcpp::streamTag& getDpcppStreamTag(const occa::streamTag& tag_)
    {
      auto* dpcppTag{dynamic_cast<occa::dpcpp::streamTag*>(tag_.getModeStreamTag())};
      OCCA_ERROR("[dpcpp::getDpcppStreamTag]: Dynamic cast failed!", nullptr != dpcppTag);
      return *dpcppTag;
    }

    void warn(const ::sycl::exception &e,
              const std::string &filename,
              const std::string &function,
              const int line,
              const std::string &message)
    {
      std::stringstream ss;
      ss << message << "\n"
         << "DPCPP Error:"
         << e.what();
      occa::warn(filename, function, line, ss.str());
    }

    void error(const ::sycl::exception &e,
               const std::string &filename,
               const std::string &function,
               const int line,
               const std::string &message)
    {
      std::stringstream ss;
      ss << message << "\n"
         << "DPCPP Error:"
         << e.what();
      occa::error(filename, function, line, ss.str());
    }

  } // namespace dpcpp
} // namespace occa
