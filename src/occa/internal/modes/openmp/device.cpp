#include <occa/internal/core/kernel.hpp>
#include <occa/internal/io/output.hpp>
#include <occa/internal/lang/modes/openmp.hpp>
#include <occa/internal/modes/serial/device.hpp>
#include <occa/internal/modes/openmp/device.hpp>
#include <occa/internal/modes/openmp/utils.hpp>

namespace occa {
  namespace openmp {
    device::device(const occa::json &properties_) :
      serial::device(properties_) {}

    hash_t device::hash() const {
      return (
        serial::device::hash()
        ^ occa::hash("openmp")
      );
    }

    hash_t device::kernelHash(const occa::json &props) const {
      return (
        serial::device::kernelHash(props)
        ^ occa::hash("openmp")
      );
    }

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

      if (!io::isFile(outputFile)) {
        hash_t hash = occa::hash(outputFile);
        io::lock_t lock(hash, "openmp-parser");
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
                                      const occa::json &kernelProps) {

      occa::json allKernelProps = properties + kernelProps;

      std::string compiler = allKernelProps["compiler"];
      int vendor = allKernelProps["vendor"];
      // Check if we need to re-compute the vendor
      if (kernelProps.has("compiler")) {
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
