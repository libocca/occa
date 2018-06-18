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

#include <omp.h>

#include <occa/modes/serial/device.hpp>
#include <occa/modes/openmp/device.hpp>
#include <occa/modes/openmp/utils.hpp>
#include <occa/lang/modes/openmp.hpp>

namespace occa {
  namespace openmp {
    device::device(const occa::properties &properties_) :
      serial::device(properties_) {
      // Generate an OpenMP library dependency (so it doesn't crash when dlclose())
      omp_get_num_threads();
    }

    bool device::parseFile(const std::string &filename,
                           const std::string &outputFile,
                           const occa::properties &parserProps) {
      lang::okl::openmpParser parser(parserProps);
      parser.parseFile(filename);

      // Verify if parsing succeeded
      if (!parser.succeeded()) {
        if (!parserProps.get("silent", false)) {
          OCCA_FORCE_ERROR("Unable to transform OKL kernel");
        }
        return false;
      }

      if (!sys::fileExists(outputFile)) {
        hash_t hash = occa::hash(outputFile);
        io::lock_t lock(hash, "serial-parser");
        if (lock.isMine()) {
          parser.writeToFile(outputFile);
        }
      }

      return true;
    }

    kernel_v* device::buildKernel(const std::string &filename,
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
        allKernelProps["compilerFlags"] += " " + lastCompilerOpenMPFlag;
      }

      kernel_v *k = serial::device::buildKernel(filename,
                                                kernelName,
                                                kernelHash,
                                                allKernelProps);

      if (k && usingOpenMP) {
        k->dHandle->removeRef();
        k->dHandle = this;
        addRef();
      }

      return k;
    }
  }
}

#endif
