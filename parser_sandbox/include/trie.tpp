#include <cstring>

#include <alloca.h>

#include "occa/defines.hpp"
#include "occa/tools/sys.hpp"

namespace occa {
  //---[ Trie ]-------------------------
  //  ---[ Result ]---------------------
  template <class TM>
  trie<TM>::result_t::result_t() {}

  template <class TM>
  trie<TM>::result_t::result_t(const trie<TM> *trie__,
                               const int length_,
                               const int valueIdx_) :
    trie_(const_cast<trie<TM>*>(trie__)),
    length(length_),
    valueIdx(valueIdx_) {}

  template <class TM>
  bool trie<TM>::result_t::success() const {
    return (0 <= valueIdx);
  }

  template <class TM>
  const TM& trie<TM>::result_t::value() const {
    return ((0 <= valueIdx)
            ? trie_->values[valueIdx]
            : trie_->defaultValue);
  }

  template <class TM>
  TM& trie<TM>::result_t::value() {
    return ((0 <= valueIdx)
            ? trie_->values[valueIdx]
            : trie_->defaultValue);
  }
  //  ==================================

  template <class TM>
  trie<TM>::trie() :
    isFrozen(false),
    autoFreeze(true),
    nodeCount(0),
    baseNodeCount(0),
    chars(NULL),
    offsets(NULL),
    leafCount(NULL),
    valueIndices(NULL) {}

  template <class TM>
  void trie<TM>::clear() {
    root.leaves.clear();
    values.clear();
    defrost();
  }

  template <class TM>
  bool trie<TM>::isEmpty() const {
    return (root.leaves.size() == 0);
  }

  template <class TM>
  void trie<TM>::add(const char *c, const TM &value) {
    defrost();
    int valueIdx = root.getValueIdx(c);
    if (valueIdx < 0) {
      valueIdx = values.size();
      values.push_back(value);
    } else {
      values[valueIdx] = value;
    }
    root.add(c, valueIdx);
    if (autoFreeze) {
      freeze();
    }
  }

  template <class TM>
  void trie<TM>::add(const std::string &s, const TM &value) {
    add(s.c_str(), value);
  }

  template <class TM>
  void trie<TM>::freeze() {
    defrost();

    nodeCount     = root.nodeCount();
    baseNodeCount = (int) root.leaves.size();

    chars        = new char[nodeCount + 1];
    offsets      = new int[nodeCount + 1];
    leafCount    = new int[nodeCount + 1];
    valueIndices = new int[nodeCount + 1];

    chars[nodeCount]        = '\0';
    offsets[nodeCount]      = nodeCount;
    leafCount[nodeCount]    = 0;
    valueIndices[nodeCount] = -1;

    freeze(root, 0);
    isFrozen = true;
  }

  template <class TM>
  int trie<TM>::freeze(const trieNode &node, int offset) {
    const trieNodeMap_t &leaves = node.leaves;
    cTrieNodeMapIterator leaf = leaves.begin();
    int leafOffset = offset + (int) leaves.size();

    while (leaf != leaves.end()) {
      const trieNode &leafNode = leaf->second;
      chars[offset]        = leaf->first;
      offsets[offset]      = leafOffset;
      leafCount[offset]    = (int) leafNode.leaves.size();
      valueIndices[offset] = leafNode.valueIdx;

      leafOffset = freeze(leafNode, leafOffset);
      ++offset;
      ++leaf;
    }
    return leafOffset;
  }

  template <class TM>
  void trie<TM>::defrost() {
    if (isFrozen) {
      nodeCount     = 0;
      baseNodeCount = 0;

      delete [] chars;
      delete [] offsets;
      delete [] leafCount;
      delete [] valueIndices;

      chars        = NULL;
      offsets      = NULL;
      leafCount    = NULL;
      valueIndices = NULL;

      isFrozen = false;
    }
  }

  template <class TM>
  typename trie<TM>::result_t trie<TM>::getLongest(const char *c,
                                                   const int length) const {
    if (!isFrozen) {
      return trieGetLongest(c, length);
    }
    const char * const cStart = c;
    int retLength = 0;
    int retValueIdx = -1;

    int offset = 0;
    int count = baseNodeCount;
    for (int i = 0; i < length; ++i) {
      const char ci = *c;
      int start = 0, end = (count - 1);
      bool found = false;
      while (start <= end) {
        // Faster than storing the value
#define OCCA_TRIE_MID ((start + end) / 2)
        const char cmid = chars[offset + OCCA_TRIE_MID];
        if (ci < cmid) {
          end = (OCCA_TRIE_MID - 1);
        } else if (ci > cmid) {
          start = (OCCA_TRIE_MID + 1);
        } else {
          ++c;
          if (0 <= valueIndices[offset + OCCA_TRIE_MID]) {
            retLength   = (c - cStart);
            retValueIdx = valueIndices[offset + OCCA_TRIE_MID];
          }

          count  = leafCount[offset + OCCA_TRIE_MID];
          offset = offsets[offset + OCCA_TRIE_MID];

          found = true;
          break;
        }
#undef OCCA_TRIE_MID
      }
      if (!found) {
        break;
      }
    }

    if (retLength) {
      return result_t(this, retLength, retValueIdx);
    }
    return result_t(this);
  }

  template <class TM>
  typename trie<TM>::result_t trie<TM>::trieGetLongest(const char *c, const int length) const {
    trieNode::result_t result = root.get(c, length);
    if (result.success()) {
      return result_t(this, result.length, result.valueIdx);
    }
    return result_t(this);
  }

  template <class TM>
  typename trie<TM>::result_t trie<TM>::get(const char *c, const int length) const {
    const int length_ = (length == INT_MAX) ? strlen(c) : length;
    trie<TM>::result_t result = getLongest(c, length_);
    if (result.length != length_) {
      result.length = 0;
      result.valueIdx = -1;
    }
    return result;
  }

  template <class TM>
  bool trie<TM>::has(const char c) const {
    if (!isFrozen) {
      return trieHas(c);
    }
    for (int i = 0; i < baseNodeCount; ++i) {
      if (chars[i] == c) {
        return true;
      }
    }
    return false;
  }

  template <class TM>
  bool trie<TM>::trieHas(const char c) const {
    const trieNodeMap_t &leaves = root.leaves;
    cTrieNodeMapIterator it = leaves.begin();
    while (it != leaves.end()) {
      if (c == it->first) {
        return true;
      }
      ++it;
    }
    return false;
  }

  template <class TM>
  bool trie<TM>::has(const char *c) const {
    result_t result = get(c);
    return ((size_t) result.length == strlen(c));
  }

  template <class TM>
  bool trie<TM>::has(const char *c, const int size) const {
    OCCA_ERROR("Cannot search for a char* with size: " << size,
               0 < size);
    return (get(c, size).success());
  }

  template <class TM>
  void trie<TM>::print() {
    const bool wasFrozen = isFrozen;
    if (!isFrozen) {
      freeze();
    }
    const std::string headers[] = {
      "index    : ",
      "chars    : ",
      "offsets  : ",
      "leafCount: ",
      "valueIdx : "
    };
    std::cout << headers[0];
    for (int i = 0; i < nodeCount; ++i) {
      std::string spaces(i < 10 ? 2 : ((i < 100) ? 1 : 0), ' ');
      std::cout << i << ' ' << spaces;
    }
    std::cout << '\n' << headers[1];
    for (int i = 0; i < nodeCount; ++i) {
      std::cout << chars[i] << "   ";
    }
    std::cout << '\n' << headers[2];
    for (int i = 0; i < nodeCount; ++i) {
      const int offset = offsets[i];
      std::string spaces(offset < 10 ? 2 : ((offset < 100) ? 1 : 0), ' ');
      std::cout << offsets[i] << ' ' << spaces;
    }
    std::cout << '\n' << headers[3];
    for (int i = 0; i < nodeCount; ++i) {
      const int lcount = leafCount[i];
      std::string spaces(lcount < 10 ? 2 : ((lcount < 100) ? 1 : 0), ' ');
      std::cout << leafCount[i] << ' ' << spaces;
    }
    std::cout << '\n' << headers[4];
    for (int i = 0; i < nodeCount; ++i) {
      const int valueIdx = valueIndices[i];
      std::string spaces(valueIdx < 10 ? 2 : ((valueIdx < 100) ? 1 : 0), ' ');
      if (valueIdx < 0) {
        spaces = spaces.substr(1);
      }
      std::cout << valueIdx << ' ' << spaces;
    }
    std::cout << '\n';
    isFrozen = wasFrozen;
  }
  //====================================
}
