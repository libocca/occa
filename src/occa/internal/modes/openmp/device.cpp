#include <occa/internal/core/kernel.hpp>
#include <occa/internal/utils/env.hpp>
#include <occa/internal/io/output.hpp>
#include <occa/internal/lang/modes/openmp.hpp>
#include <occa/internal/modes/serial/device.hpp>
#include <occa/internal/modes/openmp/device.hpp>
#include <occa/internal/modes/openmp/utils.hpp>

#ifdef BUILD_WITH_OCCA_TRANSPILER
#include <occa/internal/utils/transpiler_utils.h>
#include "oklt/pipeline/normalizer_and_transpiler.h"
#include "oklt/core/error.h"
#include "oklt/util/io_helper.h"
#endif

namespace occa {
  namespace openmp {
    device::device(const occa::json &properties_) :
      serial::device(properties_) {}

    hash_t device::hash() const {
      return (
        serial::device::hash()
        ^ occa::hash("openmp device::hash")
      );
    }

    hash_t device::kernelHash(const occa::json &props) const {
      return (
        serial::device::kernelHash(props)
        ^ occa::hash("openmp device::kernelHash")
      );
    }

#ifdef BUILD_WITH_OCCA_TRANSPILER
    bool device::transpileFile(const std::string &filename,
                       const std::string &outputFile,
                       const occa::json &kernelProps,
                       lang::sourceMetadata_t &metadata)
    {
      auto defines = transpiler::buildDefines(kernelProps);
      auto includes = transpiler::buildIncludes(kernelProps);
      auto hash = transpiler::getKernelHash(kernelProps);

      std::filesystem::path sourcePath = io::expandFilename(filename);
      auto sourceCode = oklt::util::readFileAsStr(sourcePath);
      if(!sourceCode) {
          std::string errorDescription = "Can't read file: ";
          OCCA_FORCE_ERROR(errorDescription << sourcePath.string());
          return false;
      }
      oklt::UserInput input {
          .backend = oklt::TargetBackend::OPENMP,
          .source = std::move(sourceCode.value()),
          .headers = {},
          .sourcePath = sourcePath,
          .includeDirectories = std::move(includes),
          .defines = std::move(defines),
          .hash = std::move(hash),
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
              std::filesystem::path transpiledSource(tempFilename);
              auto ret = oklt::util::writeFileAsStr(tempFilename, userOutput.kernel.source);
              return ret.has_value();
          });
      transpiler::makeMetadata(metadata, userOutput.kernel.metadata);
      return true;
    }
#endif

    bool device::parseFile(const std::string &filename,
                           const std::string &outputFile,
                           const occa::json &kernelProps,
                           lang::sourceMetadata_t &metadata) {
      lang::okl::openmpParser parser(kernelProps);
      parser.parseFile(filename);

      // Verify if parsing succeeded
      if (!parser.succeeded()) {
        if (!kernelProps.get("silent", false)) {
          OCCA_FORCE_ERROR("Unable to transform OKL kernel [" << filename << "]");
        }
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

    modeKernel_t* device::buildKernel(const std::string &filename,
                                      const std::string &kernelName,
                                      const hash_t kernelHash,
                                      const occa::json &kernelProps) {

      occa::json allKernelProps = properties + kernelProps;

      std::string compilerLanguage;
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

      std::string compiler;
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

      int vendor = allKernelProps["vendor"];
      // Check if we need to re-compute the vendor
      if (compiler.size()) {
        vendor = sys::compilerVendor(compiler);
      }

      if (compiler != lastCompiler) {
        lastCompiler = compiler;
        lastCompilerOpenMPFlag = openmp::compilerFlag(vendor, compiler);

        if (lastCompilerOpenMPFlag == openmp::notSupported) {
          io::stderr << "Compiler [" << (std::string) allKernelProps["compiler"]
                     << "] does not support OpenMP, defaulting to [Serial] mode\n";
        }
      }

      const bool usingOpenMP = (lastCompilerOpenMPFlag != openmp::notSupported);
      if (usingOpenMP) {
        allKernelProps["compiler_flags"] += " " + lastCompilerOpenMPFlag;
      }

      modeKernel_t *k = serial::device::buildKernel(filename,
                                                    kernelName,
                                                    kernelHash,
                                                    allKernelProps);

      if (k && usingOpenMP) {
        k->modeDevice->removeKernelRef(k);
        k->modeDevice = this;
        addKernelRef(k);
      }

      return k;
    }
  }
}
