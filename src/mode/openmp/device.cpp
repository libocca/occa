#include <occa/defines.hpp>

#if OCCA_OPENMP_ENABLED

#include <occa/io/output.hpp>
#include <occa/lang/mode/openmp.hpp>
#include <occa/mode/serial/device.hpp>
#include <occa/mode/openmp/device.hpp>
#include <occa/mode/openmp/utils.hpp>

namespace occa {
  namespace openmp {
    device::device(const occa::properties &properties_) :
      serial::device(properties_) {}

    hash_t device::kernelHash(const occa::properties &props) const {
      return (
        occa::hash(props["vendor"])
        ^ props["compiler"]
        ^ props["compiler_flags"]
        ^ props["compiler_env_script"]
      );
    }

    bool device::parseFile(const std::string &filename,
                           const std::string &outputFile,
                           const occa::properties &kernelProps,
                           lang::kernelMetadataMap &metadata) {
      lang::okl::openmpParser parser(kernelProps);
      parser.parseFile(filename);

      // Verify if parsing succeeded
      if (!parser.succeeded()) {
        if (!kernelProps.get("silent", false)) {
          OCCA_FORCE_ERROR("Unable to transform OKL kernel");
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

      parser.setMetadata(metadata);

      return true;
    }

    modeKernel_t* device::buildKernel(const std::string &filename,
                                      const std::string &kernelName,
                                      const hash_t kernelHash,
                                      const occa::properties &kernelProps) {

      occa::properties allKernelProps = properties + kernelProps;

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

#endif
