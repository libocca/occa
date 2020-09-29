#include <occa/core/base.hpp>
#include <occa/tools/env.hpp>
#include <occa/tools/sys.hpp>
#include <occa/modes/dpcpp/device.hpp>
#include <occa/modes/dpcpp/kernel.hpp>
#include <occa/modes/dpcpp/memory.hpp>
#include <occa/modes/dpcpp/stream.hpp>
#include <occa/modes/dpcpp/streamTag.hpp>
#include <occa/modes/dpcpp/utils.hpp>
#include <occa/modes/serial/device.hpp>
#include <occa/modes/serial/kernel.hpp>
#include <occa/lang/kernelMetadata.hpp>
#include <occa/lang/primitive.hpp>
#include <occa/lang/modes/dpcpp.hpp>

namespace occa {
  namespace dpcpp {
    device::device(const occa::properties &properties_) :
      occa::modeDevice_t(properties_) {

      if (!properties.has("wrapped")) {
        int error;
        OCCA_ERROR("[DPCPP] device not given a [platform_id] integer",
                   properties.has("platform_id") &&
                   properties["platform_id"].isNumber());

        OCCA_ERROR("[DPCPP] device not given a [device_id] integer",
                   properties.has("device_id") &&
                   properties["device_id"].isNumber());

      /*  platformID = properties.get<int>("platform_id");
        deviceID   = properties.get<int>("device_selector");
*/
        dpcppDevice = ::sycl::device();

//        dpcppContext = clCreateContext(NULL, 1, &clDevice, NULL, NULL, &error);
//        OCCA_DPCPP_ERROR("Device: Creating Context", error);
      }

      occa::json &kernelProps = properties["kernel"];
      std::string compilerFlags;

      // Use "-cl-opt-disable" for debug-mode
      if (env::var("OCCA_DPCPP_COMPILER_FLAGS").size()) {
        compilerFlags = env::var("OCCA_DPCPP_COMPILER_FLAGS");
      } else if (kernelProps.has("compiler_flags")) {
        compilerFlags = (std::string) kernelProps["compiler_flags"];
      }

      kernelProps["compiler_flags"] = compilerFlags;
    }

    device::~device() {
    }

    void device::finish() const {
      getCommandQueue()->wait();
    }

    bool device::hasSeparateMemorySpace() const {
      return true;
    }

    hash_t device::hash() const {
      if (!hash_.initialized) {
        std::stringstream ss;
        ss << "platform: " << platformID << ' '
           << "device: " << deviceID;
        hash_ = occa::hash(ss.str());
      }
      return hash_;
    }

    hash_t device::kernelHash(const occa::properties &props) const {
      return occa::hash(props["compiler_flags"]);
    }

/*
    bool device::parseFile(const std::string &filename,
                           const std::string &outputFile,
                           const occa::properties &kernelProps,
                           lang::sourceMetadata_t &metadata) {
      lang::okl::serialParser parser(kernelProps);
      parser.parseFile(filename);

      // Verify if parsing succeeded
      if (!parser.succeeded()) {
        OCCA_ERROR("Unable to transform OKL kernel",
                   kernelProps.get("silent", false));
        return false;
      }

      if (!io::isFile(outputFile)) {
        hash_t hash = occa::hash(outputFile);
        io::lock_t lock(hash, "serial-parser");
        if (lock.isMine()) {
          parser.writeToFile(outputFile);
        }
      }

      parser.setSourceMetadata(metadata);

      return true;
    }

*/
/*    lang::okl::withLauncher* device::createParser(const occa::properties &props) const {
      return new lang::okl::dpcppParser(props);
    }
*/
    //---[ Stream ]---------------------
    modeStream_t* device::createStream(const occa::properties &props) {
      ::sycl::queue* q = new ::sycl::queue();
      return new stream(this, props, q);
    }

    occa::streamTag device::tagStream() {
      return new occa::dpcpp::streamTag(this);
    }

    void device::waitFor(occa::streamTag tag) {
/*      occa::opencl::streamTag *clTag = (
        dynamic_cast<occa::opencl::streamTag*>(tag.getModeStreamTag())
      );
      OCCA_OPENCL_ERROR("Device: Waiting For Tag",
                        clWaitForEvents(1, &(clTag->clEvent)));*/
    }

    double device::timeBetween(const occa::streamTag &startTag,
                               const occa::streamTag &endTag) {
      occa::dpcpp::streamTag *dpcppStartTag = (
        dynamic_cast<occa::dpcpp::streamTag*>(startTag.getModeStreamTag())
      );
      occa::dpcpp::streamTag *dpcppEndTag = (
        dynamic_cast<occa::dpcpp::streamTag*>(endTag.getModeStreamTag())
      );

      finish();

      return (dpcppEndTag->getTime() - dpcppStartTag->getTime());
    }

    ::sycl::queue *device::getCommandQueue() const {
      occa::dpcpp::stream *stream = (occa::dpcpp::stream*) currentStream.getModeStream();
      return stream->commandQueue;
    }
    //==================================

    //---[ Kernel ]---------------------
  
   bool device::parseFile(const std::string &filename,
                           const std::string &outputFile,
                           const occa::properties &kernelProps,
                           lang::sourceMetadata_t &metadata) {
      lang::okl::serialParser parser(kernelProps);
      parser.parseFile(filename);

      // Verify if parsing succeeded
      if (!parser.succeeded()) {
        OCCA_ERROR("Unable to transform OKL kernel",
                   kernelProps.get("silent", false));
        return false;
      }

      if (!io::isFile(outputFile)) {
        hash_t hash = occa::hash(outputFile);
        io::lock_t lock(hash, "serial-parser");
        if (lock.isMine()) {
          parser.writeToFile(outputFile);
        }
      }

      parser.setSourceMetadata(metadata);

      return true;
    }

    modeKernel_t* device::buildKernel(const std::string &filename,
                                      const std::string &kernelName,
                                      const hash_t kernelHash,
                                      const occa::properties &kernelProps) {
      return buildKernel(filename, kernelName, kernelHash, kernelProps, false);
    }

    modeKernel_t* device::buildLauncherKernel(const std::string &filename,
                                              const std::string &kernelName,
                                              const hash_t kernelHash) {
      return buildKernel(filename, kernelName, kernelHash, properties["kernel"], true);
    }

    modeKernel_t* device::buildKernel(const std::string &filename,
                                      const std::string &kernelName,
                                      const hash_t kernelHash,
                                      const occa::properties &kernelProps,
                                      const bool isLauncherKernel) {
      const std::string hashDir = io::hashDir(filename, kernelHash);

      const std::string &kcBinaryFile = (
        isLauncherKernel
        ? kc::launcherBinaryFile
        : kc::binaryFile
      );
      std::string binaryFilename = hashDir + kcBinaryFile;

      // Check if binary exists and is finished
      bool foundBinary = (
        io::cachedFileIsComplete(hashDir, kcBinaryFile)
        && io::isFile(binaryFilename)
      );

      io::lock_t lock;
      if (!foundBinary) {
        lock = io::lock_t(kernelHash, "serial-kernel");
        foundBinary = !lock.isMine();
      }

      const bool verbose = kernelProps.get("verbose", false);
      if (foundBinary) {
        if (verbose) {
          io::stdout << "Loading cached ["
                     << kernelName
                     << "] from ["
                     << io::shortname(filename)
                     << "] in [" << io::shortname(binaryFilename) << "]\n";
        }
        modeKernel_t *k = buildKernelFromBinary(binaryFilename,
                                                kernelName,
                                                kernelProps);
        if (k) {
          k->sourceFilename = filename;
        }
        return k;
      }

      std::string sourceFilename;
      lang::sourceMetadata_t metadata;
      const bool compilingOkl = kernelProps.get("okl/enabled", true);
      const bool compilingCpp = (
        ((int) kernelProps["compiler_language"]) == sys::language::CPP
      );

      if (isLauncherKernel) {
        sourceFilename = filename;
      } else {
        const std::string &rawSourceFile = (
          compilingCpp
          ? kc::cppRawSourceFile
          : kc::cRawSourceFile
        );

        // Cache raw origin
        sourceFilename = (
          io::cacheFile(filename,
                        rawSourceFile,
                        kernelHash,
                        assembleKernelHeader(kernelProps))
        );

        if (compilingOkl) {
          const std::string outputFile = hashDir + kc::sourceFile;
          bool valid = parseFile(sourceFilename,
                                 outputFile,
                                 kernelProps,
                                 metadata);
          if (!valid) {
            return NULL;
          }
          sourceFilename = outputFile;

          writeKernelBuildFile(hashDir + kc::buildFile,
                               kernelHash,
                               kernelProps,
                               metadata);
        }
      }

      std::stringstream command;
      std::string compilerEnvScript = kernelProps["compiler_env_script"];
      if (compilerEnvScript.size()) {
        command << compilerEnvScript << " && ";
      }

      const std::string compiler = kernelProps["compiler"];
      std::string compilerFlags = kernelProps["compiler_flags"];
      std::string compilerLinkerFlags = kernelProps["compiler_linker_flags"];
      std::string compilerSharedFlags = kernelProps["compiler_shared_flags"];

      sys::addCompilerFlags(compilerFlags, compilerSharedFlags);

      if (!compilingOkl) {
        sys::addCompilerIncludeFlags(compilerFlags);
        sys::addCompilerLibraryFlags(compilerFlags);
      }

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
      command << compiler
              << ' '    << compilerFlags
              << ' '    << sourceFilename
              << " -o " << binaryFilename
              << " -I"  << env::OCCA_DIR << "include"
              << " -I"  << env::OCCA_INSTALL_DIR << "include"
              << " -L"  << env::OCCA_INSTALL_DIR << "lib -locca"
              << ' '    << compilerLinkerFlags
              << std::endl;
#else
      command << kernelProps["compiler"]
              << " /D MC_CL_EXE"
              << " /D OCCA_OS=OCCA_WINDOWS_OS"
              << " /EHsc"
              << " /wd4244 /wd4800 /wd4804 /wd4018"
              << ' '       << compilerFlags
              << " /I"     << env::OCCA_DIR << "include"
              << " /I"     << env::OCCA_INSTALL_DIR << "include"
              << ' '       << sourceFilename
              << " /link " << env::OCCA_INSTALL_DIR << "lib/libocca.lib",
              << ' '       << compilerLinkerFlags
              << " /OUT:"  << binaryFilename
              << std::endl;
#endif

      const std::string &sCommand = strip(command.str());

      if (verbose) {
        io::stdout << "Compiling [" << kernelName << "]\n" << sCommand << "\n";
      }

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
      const int compileError = system(sCommand.c_str());
#else
      const int compileError = system(("\"" +  sCommand + "\"").c_str());
#endif

      lock.release();
      if (compileError) {
        OCCA_FORCE_ERROR("Error compiling [" << kernelName << "],"
                         " Command: [" << sCommand << ']');
      }

      modeKernel_t *k = buildKernelFromBinary(binaryFilename,
                                              kernelName,
                                              kernelProps,
                                              metadata.kernelsMetadata[kernelName]);
      if (k) {
        io::markCachedFileComplete(hashDir, kcBinaryFile);
        k->sourceFilename = filename;
      }
      return k;
    }

    modeKernel_t* device::buildKernelFromBinary(const std::string &filename,
                                                const std::string &kernelName,
                                                const occa::properties &kernelProps) {
      std::string buildFile = io::dirname(filename);
      buildFile += kc::buildFile;

      lang::kernelMetadata_t metadata;
      if (io::isFile(buildFile)) {
        lang::sourceMetadata_t sourceMetadata = lang::sourceMetadata_t::fromBuildFile(buildFile);
        metadata = sourceMetadata.kernelsMetadata[kernelName];
      }

      return buildKernelFromBinary(filename,
                                   kernelName,
                                   kernelProps,
                                   metadata);
    }

    modeKernel_t* device::buildKernelFromBinary(const std::string &filename,
                                                const std::string &kernelName,
                                                const occa::properties &kernelProps,
                                                lang::kernelMetadata_t &metadata) {
      kernel &k = *(new kernel(this,
                               kernelName,
                               filename,
                               kernelProps));

      k.binaryFilename = filename;
      k.metadata = metadata;

      k.dlHandle = sys::dlopen(filename);
      k.function = sys::dlsym(k.dlHandle, kernelName);

      return &k;
    }


    //==================================

    //---[ Memory ]---------------------
    modeMemory_t* device::malloc(const udim_t bytes,
                                 const void *src,
                                 const occa::properties &props) {
      ::sycl::queue* q = getCommandQueue();
/*      if (props.get("mapped", false)) {
        return mappedAlloc(bytes, src, props);
      }
*/

      memory *mem = new memory(this, bytes, props);

      if (src == NULL) {
        mem->dpcppMem = malloc_device(bytes, q->get_device(), q->get_context());
      } else {
	mem->dpcppMem = malloc_device(bytes, q->get_device(), q->get_context());
	q->memcpy(mem->dpcppMem, src, bytes);
        finish();
      }
      mem->rootDpcppMem = mem->dpcppMem;
      return mem;
    }

    modeMemory_t* device::mappedAlloc(const udim_t bytes,
                                      const void *src,
                                      const occa::properties &props) {
	return malloc(bytes, src, props);
/*
      opencl::memory *mem = new opencl::memory(this, bytes, props);

      // Alloc pinned host buffer
      mem->clMem = clCreateBuffer(clContext,
                                  CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                  bytes,
                                  NULL, &error);
      mem->rootClMem = &mem->clMem;

      OCCA_OPENCL_ERROR("Device: clCreateBuffer", error);

      if (src != NULL){
        mem->copyFrom(src, mem->size);
      }

      // Map memory to read/write
      mem->mappedPtr = clEnqueueMapBuffer(getCommandQueue(),
                                          mem->clMem,
                                          CL_TRUE,
                                          CL_MAP_READ | CL_MAP_WRITE,
                                          0, bytes,
                                          0, NULL, NULL,
                                          &error);

      OCCA_OPENCL_ERROR("Device: clEnqueueMapBuffer", error);

      // Sync memory mapping
      finish();

      return mem;
*/
	    return NULL;
      }

    udim_t device::memorySize() const {
      return dpcpp::getDeviceMemorySize(dpcppDevice);
    }
    //==================================
  }
}
