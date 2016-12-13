/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2016 David Medina and Tim Warburton
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
#include "occa/parser/parser.hpp"

namespace occa {
  //---[ properties ]-------------------
  properties::properties(hasProperties *holder_) :
    holder(holder_) {}

  properties::properties(const std::string &props_) {
    if (props_.size() == 0)
      return;

    parserNS::expNode expRoot = parserNS::createOrganizedExpNodeFrom(props_);
    parserNS::expNode &csvFlatRoot = *(expRoot.makeCsvFlatHandle());

    for (int i = 0; i < csvFlatRoot.leafCount; ++i) {
      parserNS::expNode &leaf = csvFlatRoot[i];
      std::string &prop = (leaf.leafCount ? leaf[0].value : leaf.value);

      if (leaf.value != "=") {
        std::cout << "Property [" << prop << "] was not set, skipping it\n";
        continue;
      }

      set(prop, leaf[1].toString());
    }

    parserNS::expNode::freeFlatHandle(csvFlatRoot);
  }

  properties::properties(const properties &p) {
    *this = p;
  }

  properties& properties::operator = (const properties &p) {
    props = p.props;
    return *this;
  }

  bool properties::has(const std::string &prop) const {
    citer_t it = props.find(prop);
    return ((it != props.end()) && it->second.size());
  }

  std::string& properties::operator [] (const std::string &prop) {
    return props[prop];
  }

  const std::string properties::operator [] (const std::string &prop) const {
    citer_t it = props.find(prop);
    if (it != props.end()) {
      return it->second;
    }
    return "";
  }

  std::string properties::get(const std::string &prop, const std::string &default_) const {
    return get<std::string>(prop, default_);
  }

  properties properties::operator + (const properties &other) const {
    properties all = *this;
    citer_t it = other.props.begin();
    while (it != other.props.end()) {
      all.props[it->first] = it->second;
      ++it;
    }
    return all;
  }

  void properties::remove(const std::string &prop) {
    iter_t it = props.find(prop);
    std::string &oldValue = it->second;
    onChange(Remove, prop, oldValue, "");
    props.erase(it);
  }

  bool properties::iMatch(const std::string &prop, const std::string &value) const {
    citer_t it = props.find(prop);
    if (it == props.end()) {
      return false;
    }
    return (lowercase(it->second) == lowercase(value));
  }

  void properties::setOnChangeFunc(hasProperties &holder_) {
    holder = &holder_;
  }

  void properties::onChange(properties::Op op,
                            const std::string &prop,
                            const std::string &oldValue,
                            const std::string &newValue) const {
    if (holder) {
      holder->onPropertyChange(op, prop, oldValue, newValue);
    }
  }

  hash_t properties::hash() const {
    citer_t it = props.begin();
    hash_t hash_;
    while (it != props.end()) {
      hash_ ^= occa::hash(it->first);
      hash_ ^= occa::hash(it->second);
      ++it;
    }
    return hash_;
  }

  std::string properties::toString() const {
    citer_t it = props.begin();
    std::stringstream ss;

    int maxChars = 0;
    while (it != props.end()) {
      const int chars = (int) it->first.size();
      maxChars = (maxChars < chars) ? chars : maxChars;
      ++it;
    }

    ss << "{\n";

    it = props.begin();
    while (it != props.end()) {
      ss << "  "
         << it->first
         << std::string(maxChars - it->first.size() + 1, ' ') << ": "
         << it->second
         << ",\n";
      ++it;
    }

    ss << "}\n";
    return ss.str();
  }

  std::ostream& operator << (std::ostream &out, const properties &props) {
    out << props.toString();
    return out;
  }
  //====================================

  //---[ hasProperties ]----------------
  hasProperties::hasProperties() : properties(this) {}

  void hasProperties::onPropertyChange(properties::Op op,
                                       const std::string &prop,
                                       const std::string &oldValue,
                                       const std::string &newValue) const {}
  //====================================
}
