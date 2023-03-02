#include <occa/core/base.hpp>
#include <occa/internal/utils/env.hpp>
#include <occa/internal/utils/sys.hpp>
#include <occa/internal/modes/metal/device.hpp>
#include <occa/internal/modes/metal/kernel.hpp>
#include <occa/internal/modes/metal/buffer.hpp>
#include <occa/internal/modes/metal/memory.hpp>
#include <occa/internal/modes/metal/memoryPool.hpp>
#include <occa/internal/modes/metal/stream.hpp>
#include <occa/internal/modes/metal/streamTag.hpp>
#include <occa/internal/modes/serial/device.hpp>
#include <occa/internal/modes/serial/kernel.hpp>
#include <occa/internal/lang/modes/metal.hpp>

namespace occa {
  namespace metal {
    device::device(const occa::json &properties_) :
        occa::launchedModeDevice_t(properties_) {

      OCCA_ERROR("[Metal] device not given a [device_id] integer",
                 properties.has("device_id") &&
                 properties["device_id"].isNumber());

      occa::json &kernelProps = properties["kernel"];
      std::string compilerFlags;

      if (kernelProps.get<std::string>("compiler_flags").size()) {
        compilerFlags = (std::string) kernelProps["compiler_flags"];
      } else {
        compilerFlags = "-O3";
      }

      kernelProps["compiler_flags"] = compilerFlags;

      deviceID = properties.get<int>("device_id");

      metalDevice = api::metal::getDevice(deviceID);
      metalCommandQueue = metalDevice.createCommandQueue();
    }

    device::~device() {
      metalDevice.free();
    }

    bool device::hasSeparateMemorySpace() const {
      return true;
    }

    hash_t device::hash() const {
      if (!hash_.initialized) {
        std::stringstream ss;
        ss << "device: " << deviceID;
        hash_ = occa::hash(ss.str());
      }
      return hash_;
    }

    hash_t device::kernelHash(const occa::json &props) const {
      return occa::hash(props["compiler_flags"]);
    }

    lang::okl::withLauncher* device::createParser(const occa::json &props) const {
      return new lang::okl::metalParser(props);
    }

    //---[ Stream ]---------------------
    modeStream_t* device::createStream(const occa::json &props) {
      return new stream(this, props, metalCommandQueue);
    }

    modeStream_t* device::wrapStream(void* ptr, const occa::json &props) {
      OCCA_ERROR("A nullptr was passed to metal::device::wrapStream",nullptr != ptr);
      api::metal::commandQueue_t q = *static_cast<api::metal::commandQueue_t*>(ptr);
      return new stream(this, props, q, true);
    }

    occa::streamTag device::tagStream() {
      metal::stream &stream = (
        *((metal::stream*) (currentStream.getModeStream()))
      );
      return new occa::metal::streamTag(this, stream.metalCommandQueue.createEvent());
    }

    void device::waitFor(occa::streamTag tag) {
      occa::metal::streamTag *metalTag = (
        dynamic_cast<occa::metal::streamTag*>(tag.getModeStreamTag())
      );
      metalTag->metalEvent.waitUntilCompleted();
    }

    double device::timeBetween(const occa::streamTag &startTag,
                               const occa::streamTag &endTag) {
      occa::metal::streamTag *metalStartTag = (
        dynamic_cast<occa::metal::streamTag*>(startTag.getModeStreamTag())
      );
      occa::metal::streamTag *metalEndTag = (
        dynamic_cast<occa::metal::streamTag*>(endTag.getModeStreamTag())
      );

      waitFor(endTag);

      return (metalEndTag->getTime() - metalStartTag->getTime());
    }
    //==================================

    //---[ Kernel ]---------------------
    modeKernel_t* device::buildKernelFromProcessedSource(
      const hash_t kernelHash,
      const std::string &hashDir,
      const std::string &kernelName,
      const std::string &sourceFilename,
      const std::string &binaryFilename,
      const bool usingOkl,
      lang::sourceMetadata_t &launcherMetadata,
      lang::sourceMetadata_t &deviceMetadata,
      const occa::json &kernelProps
    ) {
      OCCA_ERROR("Metal kernels need to use OKL for now",
                 usingOkl);

      compileKernel(hashDir,
                    kernelName,
                    sourceFilename,
                    binaryFilename,
                    kernelProps);

      return buildOKLKernelFromBinary(kernelHash,
                                      hashDir,
                                      kernelName,
                                      sourceFilename,
                                      binaryFilename,
                                      launcherMetadata,
                                      deviceMetadata,
                                      kernelProps);
    }

    void device::compileKernel(const std::string &hashDir,
                               const std::string &kernelName,
                               const std::string &sourceFilename,
                               const std::string &binaryFilename,
                               const occa::json &kernelProps) {

      occa::json allProps = kernelProps;
      const bool verbose = allProps.get("verbose", false);

      const std::string airBinaryFilename = hashDir + "binary.air";

      //---[ Compile Air Binary ]-------
      std::stringstream command;

      int commandExitCode = 0;
      io::stageFile(
        airBinaryFilename,
        true,
        [&](const std::string &tempFilename) -> bool {
          command << "xcrun -sdk macosx metal -x metal"
                  << ' ' << allProps["compiler_flags"]
                  << ' ' << sourceFilename
                  << " -c -o " << tempFilename
                  << " 2>&1";

          const std::string airCommand = command.str();
          if (verbose) {
            io::stdout << "Compiling Air Binary [" << kernelName << "]\n" << airCommand << "\n";
          }

          std::string commandOutput;
          commandExitCode = sys::call(
            airCommand.c_str(),
            commandOutput
          );

          if (commandExitCode) {
            OCCA_FORCE_ERROR(
              "Error compiling [" << kernelName << "],"
              " Command: [" << airCommand << "] exited with code " << commandExitCode << "\n\n"
              << "Output:\n\n"
              << commandOutput << "\n"
            );
          } else if (verbose) {
              io::stdout << "Output:\n\n" << commandOutput << "\n";
          }

          io::sync(tempFilename);
          return true;
        }
      );

      if (commandExitCode) {
        return;
      }
      //================================

      //---[ Compile Metallib Command ]---
      command.str("");
      command << "xcrun -sdk macosx metallib"
              << ' ' << airBinaryFilename
              << " -o " << binaryFilename
              << " 2>&1";

      const std::string metallibCommand = command.str();
      if (verbose) {
        io::stdout << "Compiling Metallib [" << kernelName << "]\n" << metallibCommand << "\n";
      }

      std::string commandOutput;
      commandExitCode = sys::call(
        metallibCommand.c_str(),
        commandOutput
      );

      if (commandExitCode) {
        OCCA_FORCE_ERROR(
          "Error compiling [" << kernelName << "],"
          " Command: [" << metallibCommand << "] exited with code " << commandExitCode << "\n\n"
          << "Output:\n\n"
          << commandOutput << "\n"
        );
      } else if (verbose) {
          io::stdout << "Output:\n\n" << commandOutput << "\n";
      }
      //================================
    }

    modeKernel_t* device::buildOKLKernelFromBinary(const hash_t kernelHash,
                                                   const std::string &hashDir,
                                                   const std::string &kernelName,
                                                   const std::string &sourceFilename,
                                                   const std::string &binaryFilename,
                                                   lang::sourceMetadata_t &launcherMetadata,
                                                   lang::sourceMetadata_t &deviceMetadata,
                                                   const occa::json &kernelProps) {
      // Create wrapper kernel and set launcherKernel
      kernel &k = *(new kernel(this,
                               kernelName,
                               sourceFilename,
                               kernelProps));

      k.launcherKernel = buildLauncherKernel(kernelHash,
                                             hashDir,
                                             kernelName,
                                             launcherMetadata);

      // Find device kernels
      orderedKernelMetadata launchedKernelsMetadata = getLaunchedKernelsMetadata(
        kernelName,
        deviceMetadata
      );

      const int launchedKernelsCount = (int) launchedKernelsMetadata.size();
      for (int i = 0; i < launchedKernelsCount; ++i) {
        lang::kernelMetadata_t &metadata = launchedKernelsMetadata[i];

        api::metal::function_t metalFunction = (
          metalDevice.buildKernel(binaryFilename,
                                  metadata.name)
        );

        kernel *deviceKernel = new kernel(this,
                                          metadata.name,
                                          sourceFilename,
                                          metalDevice,
                                          metalFunction,
                                          kernelProps);
        deviceKernel->metadata = metadata;
        k.deviceKernels.push_back(deviceKernel);
      }

      return &k;
    }

    modeKernel_t* device::buildKernelFromBinary(const std::string &filename,
                                                const std::string &kernelName,
                                                const occa::json &kernelProps) {
      OCCA_FORCE_ERROR("Unable to build Metal kernels from binary");
      return NULL;
    }
    //==================================

    //---[ Memory ]---------------------
    modeMemory_t* device::malloc(const udim_t bytes,
                                 const void *src,
                                 const occa::json &props) {
      //create allocation
      buffer *buf = new metal::buffer(this, bytes, props);
      buf->malloc(bytes);

      //create slice
      memory *mem = new metal::memory(buf, bytes, 0);

      if (src) {
        mem->copyFrom(src, bytes, 0, props);
      }

      return mem;
    }

    modeMemory_t* device::wrapMemory(const void *ptr,
                                     const udim_t bytes,
                                     const occa::json &props) {
      //create allocation
      buffer *buf = new metal::buffer(this, bytes, props);
      buf->wrapMemory(ptr, bytes);

      return new metal::memory(buf, bytes, 0);
    }

    modeMemoryPool_t* device::createMemoryPool(const occa::json &props) {
      return new metal::memoryPool(this, props);
    }

    udim_t device::memorySize() const {
      return metalDevice.getMemorySize();
    }
    //==================================

    void* device::unwrap() {
      return static_cast<void*>(&metalDevice);
    }
  }
}
