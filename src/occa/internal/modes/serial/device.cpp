#include <occa/core/base.hpp>
#include <occa/internal/utils/env.hpp>
#include <occa/internal/io.hpp>
#include <occa/internal/utils/sys.hpp>
#include <occa/internal/modes/serial/device.hpp>
#include <occa/internal/modes/serial/kernel.hpp>
#include <occa/internal/modes/serial/buffer.hpp>
#include <occa/internal/modes/serial/memory.hpp>
#include <occa/internal/modes/serial/memoryPool.hpp>
#include <occa/internal/modes/serial/stream.hpp>
#include <occa/internal/modes/serial/streamTag.hpp>
#include <occa/internal/lang/modes/serial.hpp>

#ifdef BUILD_WITH_OCCA_TRANSPILER
#include <occa/internal/utils/transpiler_utils.h>
#include "oklt/pipeline/normalizer_and_transpiler.h"
#include "oklt/core/error.h"
#include "oklt/util/io_helper.h"
#endif

namespace occa {
  namespace serial {
    device::device(const occa::json &properties_) :
      occa::modeDevice_t(properties_) {
      // TODO: Maybe theres something more descriptive we can populate here
      arch = std::string("CPU");
    }

    bool device::hasSeparateMemorySpace() const {
      return false;
    }

    hash_t device::hash() const {
      if (!hash_.initialized) {
        hash_ = occa::hash("host");
      }
      return hash_;
    }

    hash_t device::kernelHash(const occa::json &props) const {
      return (
        occa::hash(props["compiler"])
        ^ props["compiler_flags"]
        ^ props["compiler_env_script"]
        ^ props["compiler_vendor"]
        ^ props["compiler_language"]
        ^ props["compiler_linker_flags"]
        ^ props["compiler_shared_flags"]
        ^ props["include_occa"]
        ^ props["link_occa"]
      );
    }

    //---[ Stream ]---------------------
    modeStream_t* device::createStream(const occa::json &props) {
      return new stream(this, props);
    }

    modeStream_t* device::wrapStream(void* ptr, const occa::json &props) {
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

#ifdef BUILD_WITH_OCCA_TRANSPILER
    bool device::transpileFile(const std::string &filename,
                               const std::string &outputFile,
                               const occa::json &kernelProps,
                               lang::sourceMetadata_t &metadata)
    {
      auto defines = transpiler::buildDefines(kernelProps);
      auto includes = transpiler::buildIncludes(kernelProps);

      std::filesystem::path sourcePath = io::expandFilename(filename);
      auto sourceCode = oklt::util::readFileAsStr(sourcePath);
      if(!sourceCode) {
        std::string errorDescription = "Can't read file: ";
        OCCA_FORCE_ERROR(errorDescription << sourcePath.string());
        return false;
      }
      oklt::UserInput input {
          .backend = oklt::TargetBackend::SERIAL,
          .source = std::move(sourceCode.value()),
          .headers = {},
          .sourcePath = sourcePath,
          .includeDirectories = std::move(includes),
          .defines = std::move(defines),
          .hash = "",
      };
      auto result = normalizeAndTranspile(std::move(input));
      if(!result) {
        if (!kernelProps.get("silent", false)) {
            std::stringstream ss;
            ss << "Unable to transform OKL kernel [" << filename << "]" << std::endl;
            ss << "Transpilation errors occured: " << std::endl;
            for(const auto &err: result.error()) {
                ss << err.desc << std::endl;
            }
            OCCA_FORCE_ERROR(ss.str());
        }
        return false;
      }

      auto userOutput = result.value();

      io::stageFile(
          outputFile,
          true,
          [&](const std::string &tempFilename) -> bool {
              std::filesystem::path transpiledSourcePath(tempFilename);
              auto ret = oklt::util::writeFileAsStr(transpiledSourcePath, userOutput.kernel.source);
              return ret.has_value();
          });
      transpiler::makeMetadata(metadata, userOutput.kernel.metadata);
      return true;
    }
#endif

    //==================================

    //---[ Kernel ]---------------------
    bool device::parseFile(const std::string &filename,
                           const std::string &outputFile,
                           const occa::json &kernelProps,
                           lang::sourceMetadata_t &metadata) {
      lang::okl::serialParser parser(kernelProps);
      parser.parseFile(filename);

      // Verify if parsing succeeded
      if (!parser.succeeded()) {
        OCCA_ERROR("Unable to transform OKL kernel [" << filename << "]",
                   kernelProps.get("silent", false));
        return false;
      }

      io::stageFile(
        outputFile,
        true,
        [&](const std::string &tempFilename) -> bool {
          parser.writeToFile(tempFilename);
          return true;
        }
      );

      parser.setSourceMetadata(metadata);

      return true;
    }

    // TODO: Functionally obsolete overload? kernelProps from the device will now be empty anyway.
    modeKernel_t* device::buildKernel(const std::string &filename,
                                      const std::string &kernelName,
                                      const hash_t kernelHash,
                                      const occa::json &kernelProps) {
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
                                      const occa::json &kernelProps,
                                      const bool isLauncherKernel) {
      const std::string hashDir = io::hashDir(filename, kernelHash);

      const std::string &kcBinaryFile = (
        isLauncherKernel
        ? kc::launcherBinaryFile
        : kc::binaryFile
      );
      std::string binaryFilename = hashDir + kcBinaryFile;

      // Check if binary exists and is finished
      const bool foundBinary = io::isFile(binaryFilename);

      const bool verbose = kernelProps.get("verbose", false);
      if (foundBinary) {
        if (verbose) {
          io::stdout << "Loading cached ["
                     << kernelName
                     << "] from ["
                     << filename
                     << "] in [" << binaryFilename << "]\n";
        }
        modeKernel_t *k = buildKernelFromBinary(binaryFilename,
                                                kernelName,
                                                kernelProps);
        if (k) {
          k->sourceFilename = filename;
        }
        return k;
      }

      std::string compilerLanguage;
      std::string compiler;
      std::string compilerFlags;
      std::string compilerLinkerFlags;
      std::string compilerSharedFlags;
      std::string compilerEnvScript;

      // Default to C++
      compilerLanguage = "cpp";
      if (env::var("OCCA_COMPILER_LANGUAGE").size()) {
        compilerLanguage = env::var("OCCA_COMPILER_LANGUAGE");
      } else if (kernelProps.get<std::string>("compiler_language").size()) {
        compilerLanguage = (std::string) kernelProps["compiler_language"];
      }

      const bool compilingOkl = kernelProps.get("okl/enabled", true);
      const bool compilingCpp = compilingOkl || (lowercase(compilerLanguage) != "c");
      const int compilerLanguageFlag = (
        compilingCpp
        ? sys::language::CPP
        : sys::language::C
      );

      if (compilerLanguageFlag == sys::language::CPP && env::var("OCCA_CXX").size()) {
        compiler = env::var("OCCA_CXX");
      } else if (compilerLanguageFlag == sys::language::C && env::var("OCCA_CC").size()) {
        compiler = env::var("OCCA_CC");
      } else if (kernelProps.get<std::string>("compiler").size()) {
        compiler = (std::string) kernelProps["compiler"];
      } else if (compilerLanguageFlag == sys::language::CPP && env::var("CXX").size()) {
        compiler = env::var("CXX");
      } else if (compilerLanguageFlag == sys::language::C && env::var("CC").size()) {
        compiler = env::var("CC");
      } else if (compilerLanguageFlag == sys::language::CPP) {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
        compiler = "g++";
#else
        compiler = "cl.exe";
#endif
      } else {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
        compiler = "gcc";
#else
        compiler = "cl.exe";
#endif
      }

      if (kernelProps.get<std::string>("compiler_flags").size()) {
        compilerFlags = (std::string) kernelProps["compiler_flags"];
      } else if (compilerLanguageFlag == sys::language::CPP && env::var("OCCA_CXXFLAGS").size()) {
        compilerFlags = env::var("OCCA_CXXFLAGS");
      } else if (compilerLanguageFlag == sys::language::C && env::var("OCCA_CFLAGS").size()) {
        compilerFlags = env::var("OCCA_CFLAGS");
      } else if (compilerLanguageFlag == sys::language::CPP && env::var("CXXFLAGS").size()) {
        compilerFlags = env::var("CXXFLAGS");
      } else if (compilerLanguageFlag == sys::language::C && env::var("CFLAGS").size()) {
        compilerFlags = env::var("CFLAGS");
      } else {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
        compilerFlags = "-O3";
#else
        compilerFlags = " /Ox";
#endif
      }

      const int compilerVendor = sys::compilerVendor(compiler);

      if (env::var("OCCA_COMPILER_SHARED_FLAGS").size()) {
        compilerSharedFlags = env::var("OCCA_COMPILER_SHARED_FLAGS");
      } else if (kernelProps.get<std::string>("compiler_shared_flags").size()) {
        compilerSharedFlags = (std::string) kernelProps["compiler_shared_flags"];
      } else {
        compilerSharedFlags = sys::compilerSharedBinaryFlags(compilerVendor);
      }

      if (env::var("OCCA_LDFLAGS").size()) {
        compilerLinkerFlags = env::var("OCCA_LDFLAGS");
      } else if (kernelProps.get<std::string>("compiler_linker_flags").size()) {
        compilerLinkerFlags = (std::string) kernelProps["compiler_linker_flags"];
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
#  else
        //(OCCA_VS_VERSION < 1700)
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

      if (compilerLanguageFlag == sys::language::CPP) {
        sys::addCompilerFlags(compilerFlags, sys::compilerCpp11Flags(compilerVendor));
      } else if (compilerLanguageFlag == sys::language::C) {
        sys::addCompilerFlags(compilerFlags, sys::compilerC99Flags(compilerVendor));
      }

      std::string sourceFilename;
      lang::sourceMetadata_t metadata;

      if (isLauncherKernel) {
        sourceFilename = filename;
      } else {
        // Cache raw origin
        sourceFilename = (
          io::cacheFile(filename,
                        kc::cachedRawSourceFilename(filename, compilingCpp),
                        kernelHash,
                        assembleKernelHeader(kernelProps))
        );

        if (compilingOkl) {
          const std::string outputFile = hashDir + kc::cachedSourceFilename(filename);

#ifdef BUILD_WITH_OCCA_TRANSPILER
          int transpilerVersion = kernelProps.get("transpiler-version", 2);

          bool valid = false;
          if(transpilerVersion > 2) {
              valid = transpileFile(sourceFilename,
                                      outputFile,
                                      kernelProps,
                                      metadata);
          } else {
              valid = parseFile(sourceFilename,
                                 outputFile,
                                 kernelProps,
                                 metadata);

          }

          if (!valid) {
            return nullptr;
          }
#else
          if(!parseFile(sourceFilename,
                        outputFile,
                        kernelProps,
                         metadata))
          {
            return nullptr;
          }
#endif
          sourceFilename = outputFile;

          writeKernelBuildFile(hashDir + kc::buildFile,
                               kernelHash,
                               kernelProps,
                               metadata);
        }
      }

      std::stringstream command;
      if (compilerEnvScript.size()) {
        command << compilerEnvScript << " && ";
      }

      sys::addCompilerFlags(compilerFlags, compilerSharedFlags);

      if (!compilingOkl) {
        sys::addCompilerIncludeFlags(compilerFlags);
        sys::addCompilerLibraryFlags(compilerFlags);
      }

      const bool includeOcca = kernelProps.get("kernel/include_occa", isLauncherKernel);
      const bool linkOcca    = kernelProps.get("kernel/link_occa", isLauncherKernel);

      io::stageFile(
        binaryFilename,
        true,
        [&](const std::string &tempFilename) -> bool {
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
          command << compiler
                  << ' '    << compilerFlags
                  << ' '    << sourceFilename
                  << " -o " << tempFilename;
          if (includeOcca) {
            command << " -I"  << env::OCCA_DIR << "include"
                    << " -I"  << env::OCCA_INSTALL_DIR << "include";
          }
          if (linkOcca) {
            command << " -L"  << env::OCCA_INSTALL_DIR << "lib -locca";
          }
          command << ' '    << compilerLinkerFlags
                  << " 2>&1"
                  << std::endl;
#else
          command << kernelProps["compiler"]
                  << " /D MC_CL_EXE"
                  << " /D OCCA_OS=OCCA_WINDOWS_OS"
                  << " /EHsc"
                  << " /wd4244 /wd4800 /wd4804 /wd4018"
                  << ' '       << compilerFlags;
          if (includeOcca) {
            command << " /I"     << env::OCCA_DIR << "include"
                    << " /I"     << env::OCCA_INSTALL_DIR << "include";
          }
          command << ' '       << sourceFilename;
          if (linkOcca) {
            command << " /link " << env::OCCA_INSTALL_DIR << "lib/libocca.lib";
          }
          command << ' '       << compilerLinkerFlags
                  << " /OUT:"  << tempFilename
                  << std::endl;
#endif

          const std::string &sCommand = strip(command.str());
          if (verbose) {
            io::stdout << "Compiling [" << kernelName << "]\n" << sCommand << "\n";
          }

          std::string commandOutput;
#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
          const int commandExitCode = sys::call(
            sCommand.c_str(),
            commandOutput
          );
#else
          const int commandExitCode = sys::call(
            ("\"" +  sCommand + "\"").c_str(),
            commandOutput
          );
#endif

          if (commandExitCode) {
            OCCA_FORCE_ERROR(
              "Error compiling [" << kernelName << "],"
              " Command: [" << sCommand << "]\n"
              << "Output:\n\n"
              << commandOutput << "\n"
            );
          }

          return true;
        }
      );

      io::sync(binaryFilename);

      modeKernel_t *k = buildKernelFromBinary(binaryFilename,
                                              kernelName,
                                              kernelProps,
                                              metadata.kernelsMetadata[kernelName]);
      if (k) {
        k->sourceFilename = filename;
      }
      return k;
    }

    modeKernel_t* device::buildKernelFromBinary(const std::string &filename,
                                                const std::string &kernelName,
                                                const occa::json &kernelProps) {
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
                                                const occa::json &kernelProps,
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
                                 const occa::json &props) {
      //create allocation
      buffer *buf = new serial::buffer(this, bytes, props);

      if (src && props.get("use_host_pointer", false)) {
        buf->wrapMemory(src, bytes);
      } else {
        buf->malloc(bytes);
      }

      //create slice
      memory *mem = new serial::memory(buf, bytes, 0);

      if (src && !props.get("use_host_pointer", false)) {
        mem->copyFrom(src, bytes, 0, props);
      }

      return mem;
    }

    modeMemory_t* device::wrapMemory(const void *ptr,
                                     const udim_t bytes,
                                     const occa::json &props) {

      //create allocation
      buffer *buf = new serial::buffer(this, bytes, props);
      buf->wrapMemory(ptr, bytes);

      return new serial::memory(buf, bytes, 0);
    }

    modeMemoryPool_t* device::createMemoryPool(const occa::json &props) {
      return new serial::memoryPool(this, props);
    }

    udim_t device::memorySize() const {
      return sys::SystemInfo::load().memory.total;
    }
    //==================================

    void* device::unwrap() {
      OCCA_FORCE_ERROR("device::unwrap is not defined for serial mode");
      return nullptr;
    }
  }
}
