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

#ifndef OCCA_LANG_KERNELMETADATA_HEADER
#define OCCA_LANG_KERNELMETADATA_HEADER

#include <occa/tools/json.hpp>

namespace occa {
  namespace lang {
    class kernelMetadata;

    typedef std::map<std::string, kernelMetadata> kernelMetadataMap;

    class argumentInfo {
    public:
      bool isConst;

      argumentInfo(const bool isConst_ = false);

      static argumentInfo fromJson(const json &j);
      json toJson() const;
    };

    class kernelMetadata {
    public:
      std::string name;
      std::vector<argumentInfo> arguments;

      kernelMetadata();

      kernelMetadata& operator += (const argumentInfo &argInfo);

      bool argIsConst(const int pos) const;

      static kernelMetadata fromJson(const json &j);
      json toJson() const;
    };

    kernelMetadataMap getBuildFileMetadata(const std::string &filename);
  }
}

#endif
