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
    /* Returns true if any DPC++ device is enabled on the machine */
    bool isEnabled()
    {
      auto device_list = ::sycl::device::get_devices();
      return (device_list.size() > 0);
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
        compiler = "clang++";
      }
      dpcpp_properties["compiler"] = compiler;
    }

    void setCompilerFlags(occa::json &dpcpp_properties) noexcept
    {
      std::string compiler_flags;
      if (dpcpp_properties.has("compiler_flags"))
      {
        compiler_flags = dpcpp_properties["compiler_flags"].toString();
      }
      else if (env::var("OCCA_DPCPP_COMPILER_FLAGS").size())
      {
        compiler_flags = env::var("OCCA_DPCPP_COMPILER_FLAGS");
      }
      else
      {
        compiler_flags = "-O3 -fsycl";
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

    occa::device wrapDevice(::sycl::device device,
                            const occa::properties &props)
    {
      occa::properties allProps;
      allProps["mode"] = "dpcpp";
      allProps["device_id"] = -1;
      allProps["platform_id"] = -1;
      allProps["wrapped"] = true;
      allProps += props;

      auto* wrapper{new dpcpp::device(allProps)};
      wrapper->dontUseRefs();

      wrapper->dpcppDevice = device;
      wrapper->dpcppContext = ::sycl::context(device);
      wrapper->currentStream = wrapper->createStream(allProps["stream"]);

      return occa::device(wrapper);
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
