#include <stdio.h>

#include <occa/internal/modes/dpcpp/utils.hpp>
#include <occa/internal/modes/dpcpp/device.hpp>
#include <occa/internal/modes/dpcpp/memory.hpp>
#include <occa/internal/modes/dpcpp/kernel.hpp>
#include <occa/internal/modes/dpcpp/streamTag.hpp>
#include <occa/internal/io.hpp>
#include <occa/internal/utils/sys.hpp>
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
      return (::sycl::platform::get_platforms()[pID]);
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
      return ::sycl::platform::get_platforms()[pID].get_devices(deviceType(type)).size();
    }
    /* Return the DPC++ device given the platform ID and Device ID */
    ::sycl::device getDeviceByID(int pID, int dID, int type)
    {
      return (::sycl::platform::get_platforms()[pID].get_devices(deviceType(type))[dID]);
    }
    /* Return the DPC++ device name */
    std::string deviceName(int pID, int dID)
    {
      return ::sycl::platform::get_platforms()[pID].get_devices()[dID].get_info<::sycl::info::device::name>();
    }
    /* Return the DPC++ device type */
    int deviceType(int pID, int dID)
    {
      return (int)::sycl::platform::get_platforms()[pID].get_devices()[dID].get_info<::sycl::info::device::device_type>();
    }

    /* Return the DPC++ device vendor */
    int deviceVendor(int pID, int dID)
    {
      std::string devVendor = ::sycl::platform::get_platforms()[pID].get_devices()[dID].get_info<::sycl::info::device::vendor>();
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
      return ::sycl::platform::get_platforms()[pID].get_devices()[dID].get_info<::sycl::info::device::max_compute_units>();
    }
    /* Returns the DPC++ global memory size given the DPC++ device */
    udim_t getDeviceMemorySize(const ::sycl::device &devPtr)
    {
      return devPtr.get_info<::sycl::info::device::global_mem_size>();
    }
    /* Returns the DPC++ global memory size given the platform and device IDs */
    udim_t getDeviceMemorySize(int pID, int dID)
    {
      return ::sycl::platform::get_platforms()[pID].get_devices()[dID].get_info<::sycl::info::device::global_mem_size>();
    }

    // void buildProgramFromSource(info_t &info,
    //                             const std::string &source,
    //                             const std::string &kernelName,
    //                             const std::string &compilerFlags,
    //                             const std::string &sourceFile,
    //                             const occa::properties &properties,
    //                             const io::lock_t &lock) {}

    // void buildProgramFromBinary(info_t &info,
    //                             const std::string &binaryFilename,
    //                             const std::string &kernelName,
    //                             const std::string &compilerFlags,
    //                             const io::lock_t &lock) {}

    // void buildProgram(info_t &info,
    //                   const std::string &kernelName,
    //                   const std::string &compilerFlags,
    //                   const io::lock_t &lock) {}

    // void buildKernelFromProgram(info_t &info,
    //                             const std::string &kernelName,
    //                             const io::lock_t &lock) {}

    // bool saveProgramBinary(info_t &info,
    //                        const std::string &binaryFile,
    //                        const io::lock_t &lock) { return true; }

    void warn(cl_int errorCode,
              const std::string &filename,
              const std::string &function,
              const int line,
              const std::string &message)
    {
      if (!errorCode)
      {
        return;
      }

      occa::warn(filename, function, line, "dpcpp warning!");
    }

    void error(cl_int errorCode,
               const std::string &filename,
               const std::string &function,
               const int line,
               const std::string &message)
    {
      if (!errorCode)
      {
        return;
      }

      occa::error(filename, function, line, "dpcpp error!");
    }

  } // namespace dpcpp
} // namespace occa
