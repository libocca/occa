#include <occa/core/base.hpp>
#include <occa/tools/env.hpp>
#include <occa/io.hpp>
#include <occa/tools/sys.hpp>
#include <occa/modes/serial/device.hpp>
#include <occa/modes/serial/kernel.hpp>
#include <occa/modes/serial/memory.hpp>
#include <occa/modes/serial/stream.hpp>
#include <occa/modes/serial/streamTag.hpp>
#include <occa/lang/modes/serial.hpp>

namespace occa {
  namespace serial {
    device::device(const occa::properties &properties_) :
      occa::modeDevice_t(properties_) {

      occa::json &kernelProps = properties["kernel"];
      std::string compiler, compilerFlags, compilerEnvScript;
      int vendor;

      if (kernelProps.get<std::string>("compiler").size()) {
        compiler = (std::string) kernelProps["compiler"];
      } else if (env::var("OCCA_CXX").size()) {
        compiler = env::var("OCCA_CXX");
      } else if (env::var("CXX").size()) {
        compiler = env::var("CXX");
      } else {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
        compiler = "g++";
#else
        compiler = "cl.exe";
#endif
      }

      vendor = sys::compilerVendor(compiler);

      if (kernelProps.get<std::string>("compiler_flags").size()) {
        compilerFlags = (std::string) kernelProps["compiler_flags"];
      } else if (env::var("OCCA_CXXFLAGS").size()) {
        compilerFlags = env::var("OCCA_CXXFLAGS");
      } else if (env::var("CXXFLAGS").size()) {
        compilerFlags = env::var("CXXFLAGS");
      } else {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
        compilerFlags = "-O3";
#else
        compilerFlags = " /Ox";
#endif
      }

      if (kernelProps.get<std::string>("compiler_env_script").size()) {
        compilerEnvScript = (std::string) kernelProps["compiler_env_script"];
      } else {
#if (OCCA_OS == OCCA_WINDOWS_OS)
        std::string byteness;

        if (sizeof(void*) == 4) {
          byteness = "x86 ";
        } else if (sizeof(void*) == 8) {
          byteness = "amd64";
        } else {
          OCCA_FORCE_ERROR("sizeof(void*) is not equal to 4 or 8");
        }
#  if   (OCCA_VS_VERSION == 1800)
        // MSVC++ 12.0 - Visual Studio 2013
        char *visualStudioTools = getenv("VS120COMNTOOLS");
#  elif (OCCA_VS_VERSION == 1700)
        // MSVC++ 11.0 - Visual Studio 2012
        char *visualStudioTools = getenv("VS110COMNTOOLS");
#  else (OCCA_VS_VERSION < 1700)
        // MSVC++ 10.0 - Visual Studio 2010
        char *visualStudioTools = getenv("VS100COMNTOOLS");
#  endif

        if (visualStudioTools) {
          compilerEnvScript = "\"" + std::string(visualStudioTools) + "..\\..\\VC\\vcvarsall.bat\" " + byteness;
        } else {
          io::stdout << "WARNING: Visual Studio environment variable not found -> compiler environment (vcvarsall.bat) maybe not correctly setup." << std::endl;
        }
#endif
      }

      kernelProps["vendor"] = vendor;
      kernelProps["compiler"] = compiler;
      kernelProps["compiler_flags"] = compilerFlags;
      kernelProps["compiler_env_script"] = compilerEnvScript;
    }

    device::~device() {}

    void device::finish() const {}

    bool device::hasSeparateMemorySpace() const {
      return false;
    }

    hash_t device::hash() const {
      if (!hash_.initialized) {
        hash_ = occa::hash("host");
      }
      return hash_;
    }

    hash_t device::kernelHash(const occa::properties &props) const {
      return (
        occa::hash(props["vendor"])
        ^ props["compiler"]
        ^ props["compiler_flags"]
        ^ props["compiler_env_script"]
      );
    }

    //---[ Stream ]---------------------
    modeStream_t* device::createStream(const occa::properties &props) {
      return new stream(this, props);
    }

    occa::streamTag device::tagStream() {
      return new occa::serial::streamTag(this, sys::currentTime());
    }

    void device::waitFor(occa::streamTag tag) {}

    double device::timeBetween(const occa::streamTag &startTag,
                               const occa::streamTag &endTag) {
      occa::serial::streamTag *srStartTag = (
        dynamic_cast<occa::serial::streamTag*>(startTag.getModeStreamTag())
      );
      occa::serial::streamTag *srEndTag = (
        dynamic_cast<occa::serial::streamTag*>(endTag.getModeStreamTag())
      );

      return (srEndTag->time - srStartTag->time);
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
      if (isLauncherKernel) {
        sourceFilename = filename;
      } else {
        // Cache raw origin
        sourceFilename = (
          io::cacheFile(filename,
                        kc::rawSourceFile,
                        kernelHash,
                        assembleKernelHeader(kernelProps))
        );

        if (kernelProps.get("okl", true)) {
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

      std::string compilerFlags = kernelProps["compiler_flags"];
      const int vendor = (int) kernelProps["vendor"];

      sys::addSharedBinaryFlags(vendor, compilerFlags);
      sys::addCpp11Flags(vendor, compilerFlags);

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
      command << (std::string) kernelProps["compiler"]
              << ' '    << compilerFlags
              << ' '    << sourceFilename
              << " -o " << binaryFilename
              << " -I"  << env::OCCA_DIR << "include"
              << " -L"  << env::OCCA_DIR << "lib -locca"
              << std::endl;
#else
      command << kernelProps["compiler"]
              << " /D MC_CL_EXE"
              << " /D OCCA_OS=OCCA_WINDOWS_OS"
              << " /EHsc"
              << " /wd4244 /wd4800 /wd4804 /wd4018"
              << ' '       << kernelProps["compiler_flags"]
              << " /I"     << env::OCCA_DIR << "include"
              << ' '       << sourceFilename
              << " /link " << env::OCCA_DIR << "lib/libocca.lib",
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

    //---[ Memory ]-------------------
    modeMemory_t* device::malloc(const udim_t bytes,
                                 const void *src,
                                 const occa::properties &props) {
      memory *mem = new memory(this, bytes, props);

      if (src && props.get("use_host_pointer", false)) {
        mem->ptr = (char*) const_cast<void*>(src);
        mem->isOrigin = props.get("own_host_pointer", false);
      } else {
        mem->ptr = (char*) sys::malloc(bytes);
        if (src) {
          ::memcpy(mem->ptr, src, bytes);
        }
      }

      return mem;
    }

    udim_t device::memorySize() const {
      return sys::installedRAM();
    }
    //==================================
  }
}
