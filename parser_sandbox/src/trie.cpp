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
#include "trie.hpp"

namespace occa {
  //---[ Node ]-------------------------
  //  ---[ Result ]---------------------
  trieNode::result_t::result_t() {}

  trieNode::result_t::result_t(const int length_, const int valueIdx_) :
    length(length_),
    valueIdx(valueIdx_) {}
  //  ==================================

  trieNode::trieNode() :
    valueIdx(-1) {}

  trieNode::trieNode(const int valueIdx_) :
    valueIdx(valueIdx_) {}

  void trieNode::add(const char *c, const int valueIdx_) {
    if (*c == '\0') {
      valueIdx = valueIdx_;
      return;
    }
    trieNode &newNode = leaves[*c];
    newNode.add(c + 1, valueIdx_);
  }

  int trieNode::nodeCount() const {
    cTrieNodeMapIterator it = leaves.begin();
    int count = (int) leaves.size();
    while (it != leaves.end()) {
      count += it->second.nodeCount();
      ++it;
    }
    return count;
  }

  int trieNode::getValueIdx(const char *c) const {
    const int length = (int) strlen(c);
    result_t result = get(c, length);
    return (result.length == length) ? result.valueIdx : -1;
  }

  trieNode::result_t trieNode::get(const char *c, const int length) const {
    return get(c, 0, length);
  }

  trieNode::result_t trieNode::get(const char *c, const int cIdx, const int length) const {
    cTrieNodeMapIterator it = leaves.find(c[cIdx]);
    if ((cIdx < length) && (it != leaves.end())) {
      result_t result = it->second.get(c, cIdx + 1, length);
      if (!result.success() && (0 <= valueIdx)) {
        return result_t(cIdx + 1, valueIdx);
      }
      return result;
    }
    return result_t(cIdx, valueIdx);
  }
  //====================================
}
