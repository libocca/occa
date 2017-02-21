/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2017 David Medina and Tim Warburton
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

#include "occa/tools/properties.hpp"
#include "occa/tools/string.hpp"
#include "occa/parser/parser.hpp"

namespace occa {
  properties::properties() {
    type = object_;
  }

  properties::properties(const char *c) {
    load(c);
  }

  properties::properties(const std::string &s) {
    load(s);
  }

  bool properties::isInitialized() {
    return (0 < value_.object.size());
  }

  void properties::load(const char *&c) {
    loadObject(c);
  }

  void properties::load(const std::string &s) {
    const char *c = s.c_str();
    loadObject(c);
  }

  properties properties::operator + (const properties &j) const {
    properties all = *this;
    cJsonObjectIterator it = j.value_.object.begin();
    while (it != j.value_.object.end()) {
      all.value_.object[it->first] = it->second;
      ++it;
    }
    return all;
  }
}
