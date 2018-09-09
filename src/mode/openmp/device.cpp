/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 David Medina and Tim Warburton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */

#include <occa/defines.hpp>

#if OCCA_OPENMP_ENABLED

#include <occa/mode/serial/device.hpp>
#include <occa/mode/openmp/device.hpp>
#include <occa/mode/openmp/utils.hpp>
#include <occa/lang/mode/openmp.hpp>

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
          std::cerr << "Compiler [" << (std::string) allKernelProps["compiler"]
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
