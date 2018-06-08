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
#include <occa/tools/trie.hpp>

namespace occa {
  //---[ Node ]-------------------------
  //  ---[ Result ]---------------------
  trieNode::result_t::result_t() {}

  trieNode::result_t::result_t(trieNode *node_,
                               const int length_,
                               const int valueIndex_) :
    node(node_),
    length(length_),
    valueIndex(valueIndex_) {}
  //  ==================================

  trieNode::trieNode() :
    valueIndex(-1) {}

  trieNode::trieNode(const int valueIndex_) :
    valueIndex(valueIndex_) {}

  int trieNode::size() const {
    int count = (valueIndex >= 0);
    cTrieNodeMapIterator it = leaves.begin();
    while (it != leaves.end()) {
      count += it->second.size();
      ++it;
    }
    return count;
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

  int trieNode::getValueIndex(const char *c) const {
    const int length = (int) strlen(c);
    result_t result = get(c, length);
    return (result.length == length) ? result.valueIndex : -1;
  }

  void trieNode::add(const char *c, const int valueIndex_) {
    if (*c == '\0') {
      valueIndex = valueIndex_;
      return;
    }
    trieNode &newNode = leaves[*c];
    newNode.add(c + 1, valueIndex_);
  }

  trieNode::result_t trieNode::get(const char *c, const int length) const {
    return get(c, 0, length);
  }

  trieNode::result_t trieNode::get(const char *c, const int cIndex, const int length) const {
    cTrieNodeMapIterator it = leaves.find(c[cIndex]);
    if ((cIndex < length) && (it != leaves.end())) {
      result_t result = it->second.get(c, cIndex + 1, length);
      if (!result.success() && (0 <= valueIndex)) {
        return result_t(const_cast<trieNode*>(this),
                        cIndex + 1,
                        valueIndex);
      }
      return result;
    }
    return result_t(const_cast<trieNode*>(this),
                    cIndex,
                    valueIndex);
  }

  void trieNode::remove(const char *c,
                        const int valueIndex_) {
    remove(c, (int) strlen(c), valueIndex_);
  }

  void trieNode::remove(const char *c,
                        const int length,
                        const int valueIndex_) {
    if ((*c == '\0') || (length <= 0)) {
      return;
    }
    nestedRemove(c, length, valueIndex_);
    decrementIndex(valueIndex_);
  }

  bool trieNode::nestedRemove(const char *c,
                              const int length,
                              const int valueIndex_) {
    trieNodeMapIterator it = leaves.find(*c);
    if (it == leaves.end()) {
      return false;
    }

    trieNode &leaf = it->second;
    if (length > 1) {
      bool emptyTree = leaf.nestedRemove(c + 1,
                                         length - 1,
                                         valueIndex_);
      if (emptyTree && (leaves.size() == 1)) {
        leaves.erase(it);
      }
    } else {
      leaf.valueIndex = -1;
      if (!leaf.leaves.size()) {
        leaves.erase(it);
      }
    }
    return ((valueIndex < 0) && !leaves.size());
  }

  void trieNode::decrementIndex(const int valueIndex_) {
    trieNodeMapIterator it = leaves.begin();
    while (it != leaves.end()) {
      trieNode &leaf = it->second;
      if (leaf.valueIndex > valueIndex_) {
        --leaf.valueIndex;
      }
      leaf.decrementIndex(valueIndex_);
      ++it;
    }
  }
  //====================================
}
