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

#include "occa/defines.hpp"

#if OCCA_OPENMP_ENABLED

#include <omp.h>

#include "occa/modes/serial/device.hpp"
#include "occa/modes/openmp/device.hpp"
#include "occa/modes/openmp/kernel.hpp"
#include "occa/modes/openmp/utils.hpp"

namespace occa {
  namespace openmp {
    device::device(const occa::properties &properties_) :
      serial::device(properties_) {
      // Generate an OpenMP library dependency (so it doesn't crash when dlclose())
      omp_get_num_threads();
    }

    kernel_v* device::buildKernel(const std::string &filename,
                                  const std::string &kernelName,
                                  const hash_t kernelHash,
                                  const occa::properties &props) {

      occa::properties kernelProps = props;
      kernel_v *k;

      const std::string &compiler = kernelProps["compiler"].string();
      if (compiler != lastCompiler) {
        lastCompiler = compiler;

        lastCompilerOpenMPFlag = openmp::compilerFlag(
          kernelProps.get<int>("vendor"),
          compiler
        );

        if (lastCompilerOpenMPFlag == openmp::notSupported) {
          std::cerr << "Compiler [" << kernelProps["compiler"].string()
                    << "] does not support OpenMP, defaulting to [Serial] mode\n";
        }
      }

      if (lastCompilerOpenMPFlag != openmp::notSupported) {
        std::string &compilerFlags = kernelProps["compilerFlags"].string();
        compilerFlags += ' ';
        compilerFlags += lastCompilerOpenMPFlag;

        k = new kernel(kernelProps);
      } else {
        k = new serial::kernel(kernelProps);
      }
      k->setDHandle(this);
      k->build(filename, kernelName, kernelHash);
      return k;
    }
  }
}

#endif
