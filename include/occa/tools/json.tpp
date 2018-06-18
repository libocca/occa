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

namespace occa {
  template <class TM>
  TM json::get(const char *c,
               const TM &default_) const {
    const json *j = this;
    while (*c != '\0') {
      if (j->type != object_) {
        return default_;
      }

      const char *cStart = c;
      lex::skipTo(c, '/');
      std::string key(cStart, c - cStart);
      if (*c == '/') {
        ++c;
      }

      jsonObject::const_iterator it = j->value_.object.find(key);
      if (it == j->value_.object.end()) {
        return default_;
      }
      j = &(it->second);
    }
    return *j;
  }

  template <class TM>
  TM json::get(const std::string &s,
               const TM &default_) const {
    return get<TM>(s.c_str(), default_);
  }

  template <class TM>
  std::vector<TM> json::getArray(const std::vector<TM> &default_) const {
    std::string empty;
    return getArray(empty.c_str(), default_);
  }

  template <class TM>
  std::vector<TM> json::getArray(const char *c,
                                 const std::vector<TM> &default_) const {
    const json *j = this;
    while (*c) {
      if (j->type != object_) {
        return default_;
      }

      const char *cStart = c;
      lex::skipTo(c, '/');
      std::string key(cStart, c - cStart);
      if (*c == '/') {
        ++c;
      }

      jsonObject::const_iterator it = j->value_.object.find(key);
      if (it == j->value_.object.end()) {
        return default_;
      }
      j = &(it->second);
    }
    if (j->type != array_) {
      return default_;
    }

    const int entries = (int) j->value_.array.size();
    std::vector<TM> ret;
    for (int i = 0; i < entries; ++i) {
      ret.push_back((TM) j->value_.array[i]);
    }
    return ret;
  }

  template <class TM>
  std::vector<TM> json::getArray(const std::string &s,
                                 const std::vector<TM> &default_) const {
    return get<TM>(s.c_str(), default_);
  }
}
