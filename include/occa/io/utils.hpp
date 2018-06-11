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

#ifndef OCCA_IO_UTILS_HEADER
#define OCCA_IO_UTILS_HEADER

#include <iostream>

#include <occa/types.hpp>

namespace occa {
  // Kernel Caching
  namespace kc {
    extern const std::string rawSourceFile;
    extern const std::string sourceFile;
    extern const std::string binaryFile;
    extern const std::string buildFile;
    extern const std::string hostSourceFile;
    extern const std::string hostBinaryFile;
    extern const std::string hostBuildFile;
  }

  namespace io {
    const std::string& cachePath();

    const std::string& libraryPath();

    void endWithSlash(std::string &dir);

    std::string endWithSlash(const std::string &dir);

    void removeEndSlash(std::string &dir);

    std::string removeEndSlash(const std::string &dir);

    bool isAbsolutePath(const std::string &filename);

    std::string convertSlashes(const std::string &filename);

    std::string filename(const std::string &filename,
                         bool makeAbsolute = true);

    std::string binaryName(const std::string &filename);

    std::string basename(const std::string &filename,
                         const bool keepExtension = true);

    std::string dirname(const std::string &filename);

    std::string extension(const std::string &filename);

    std::string shortname(const std::string &filename);

    strVector filesInDir(const std::string &dir,
                         const unsigned char fileType);

    strVector directories(const std::string &dir);

    strVector files(const std::string &dir);

    bool exists(const std::string &filename);

    char* c_read(const std::string &filename,
                 size_t *chars = NULL,
                 const bool readingBinary = false);

    std::string read(const std::string &filename,
                     const bool readingBinary = false);

    void write(const std::string &filename,
               const std::string &content);

    std::string removeSlashes(const std::string &str);
  }
}

#endif
