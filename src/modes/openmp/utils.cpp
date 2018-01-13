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

#include <iostream>

#include "occa/tools/env.hpp"
#include "occa/tools/io.hpp"
#include "occa/tools/sys.hpp"

namespace occa {
  namespace openmp {
    std::string notSupported = "N/A";

    std::string baseCompilerFlag(const int vendor_) {
      if (vendor_ & (sys::vendor::GNU |
                     sys::vendor::LLVM)) {

        return "-fopenmp";
      } else if (vendor_ & (sys::vendor::Intel |
                            sys::vendor::Pathscale)) {

        return "-openmp";
      } else if (vendor_ & sys::vendor::IBM) {
        return "-qsmp";
      } else if (vendor_ & sys::vendor::PGI) {
        return "-mp";
      } else if (vendor_ & sys::vendor::HP) {
        return "+Oopenmp";
      } else if (vendor_ & sys::vendor::VisualStudio) {
        return "/openmp";
      } else if (vendor_ & sys::vendor::Cray) {
        return "";
      }

      return openmp::notSupported;
    }

    std::string compilerFlag(const int vendor_,
                             const std::string &compiler) {

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_OSX_OS))
      const std::string safeCompiler = io::removeSlashes(compiler);
      std::string flag = openmp::notSupported;
      std::stringstream ss;

      const std::string openmpTest = env::OCCA_DIR + "/scripts/openmpTest.cpp";
      hash_t hash = occa::hashFile(openmpTest);
      hash ^= occa::hash(vendor_);
      hash ^= occa::hash(compiler);

      const std::string srcFilename = io::cacheFile(openmpTest, "openmpTest.cpp", hash);
      const std::string binaryFilename = io::dirname(srcFilename) + "binary";
      const std::string outFilename = io::dirname(srcFilename) + "output";

      const std::string hashTag = "openmp-compiler";
      if (!io::haveHash(hash, hashTag)) {
        io::waitForHash(hash, hashTag);
      } else {
        if (!sys::fileExists(outFilename)) {
          flag = baseCompilerFlag(vendor_);
          ss << compiler
             << ' '    << flag
             << ' '    << srcFilename
             << " -o " << binaryFilename
             << " > /dev/null 2>&1";

          const std::string compileLine = ss.str();
          const int compileError = system(compileLine.c_str());

          if (compileError) {
            flag = openmp::notSupported;
          }

          io::write(outFilename, flag);
          io::releaseHash(hash, hashTag);

          return flag;
        }
        io::releaseHash(hash, hashTag);
      }

      ss << io::read(outFilename);
      ss >> flag;

      return flag;
#elif (OCCA_OS == OCCA_WINDOWS_OS)
      return "/openmp"; // VS Compilers support OpenMP
#endif
    }
  }
}

#endif
