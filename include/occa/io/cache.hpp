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

#ifndef OCCA_TOOLS_IO_CACHE_HEADER
#define OCCA_TOOLS_IO_CACHE_HEADER

#include <iostream>

#include <occa/tools/hash.hpp>

namespace occa {
  class properties;

  namespace io {
    bool isCached(const std::string &filename);

    void cache(const std::string &filename,
               std::string source,
               const hash_t &hash);

    void cache(const std::string &filename,
               const char *source,
               const hash_t &hash,
               const bool deleteSource = true);

    std::string cacheFile(const std::string &filename,
                          const std::string &cachedName,
                          const std::string &header = "");

    std::string cacheFile(const std::string &filename,
                          const std::string &cachedName,
                          const hash_t &hash,
                          const std::string &header = "");

    void writeBuildFile(const std::string &filename,
                        const hash_t &hash,
                        const occa::properties &props);

    std::string getLibraryName(const std::string &filename);

    std::string hashFrom(const std::string &filename);

    std::string hashDir(const hash_t &hash);

    std::string hashDir(const std::string &filename,
                        const hash_t &hash = hash_t());
  }
}

#endif
