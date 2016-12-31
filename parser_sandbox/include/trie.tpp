#include <cstring>

#include <alloca.h>

#include "occa/defines.hpp"
#include "occa/tools/sys.hpp"

namespace occa {
  //---[ Trie ]-------------------------
  //  ---[ Result ]---------------------
  template <class TM>
  trie_t<TM>::result_t::result_t() {}

  template <class TM>
  trie_t<TM>::result_t::result_t(const int length_, const TM &value_) :
    length(length_),
    value(value_) {}

  template <class TM>
  trie_t<TM>::result_t::operator bool() {
    return (length != 0);
  }
  //  ==================================

  template <class TM>
  trie_t<TM>::trie_t() :
    isFrozen(false),
    nodeCount(0),
    baseNodeCount(0),
    chars(NULL),
    offsets(NULL),
    leafCount(NULL),
    valueIndices(NULL) {}

  template <class TM>
  void trie_t<TM>::add(const char *c, const TM &value) {
    defrost();
    int valueIdx = root.getValueIdx(c);
    if (valueIdx < 0) {
      valueIdx = values.size();
      values.push_back(value);
    } else {
      values[valueIdx] = value;
    }
    root.add(c, valueIdx);
  }

  template <class TM>
  void trie_t<TM>::add(const std::string &s, const TM &value) {
    defrost();
    root.add(s.c_str(), value);
  }

  template <class TM>
  void trie_t<TM>::freeze() {
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
  int trie_t<TM>::freeze(const trieNode_t &node, int offset) {
    const trieNodeMap_t &leaves = node.leaves;
    cTrieNodeMapIterator leaf = leaves.begin();
    int leafOffset = offset + (int) leaves.size();

    while (leaf != leaves.end()) {
      const trieNode_t &leafNode = leaf->second;
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
  void trie_t<TM>::defrost() {
    if (isFrozen) {
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
  typename trie_t<TM>::result_t trie_t<TM>::getFirst(const char *c) const {
    if (!isFrozen) {
      return trieGetFirst(c);
    }
    const char *cStart = c;
    int retLength = 0;
    int retValueIdx = -1;

    int offset = 0;
    int count = baseNodeCount;
    while (count) {
      const char ci = *c;
      bool found = false;
      for (int i = 0; i < count; ++i) {
        if (ci == chars[offset + i]) {
          ++c;
          if (0 <= valueIndices[offset]) {
            retLength   = (c - cStart);
            retValueIdx = valueIndices[offset];
          }

          count  = leafCount[offset + i];
          offset = offsets[offset + i];

          found = true;
          break;
        }
      }
      if (!found) {
        break;
      }
    }

    if (retLength) {
      return result_t(retLength, values[retValueIdx]);
    }
    return result_t(0);
  }

  template <class TM>
  typename trie_t<TM>::result_t trie_t<TM>::trieGetFirst(const char *c) const {
    trieNode_t::result_t result = root.get(c);

    if (0 <= result.valueIdx) {
      return result_t(result.length, values[result.valueIdx]);
    }
    return result_t(0);
  }

  template <class TM>
  typename trie_t<TM>::result_t trie_t<TM>::get(const char *c) const {
    trie_t<TM>::result_t result = getFirst(c);
    if (strlen(c) != (size_t) result.length) {
      result.length = 0;
      result.value  = TM();
    }
    return result;
  }

  template <class TM>
  bool trie_t<TM>::has(const char c) const {
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
  bool trie_t<TM>::trieHas(const char c) const {
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
  bool trie_t<TM>::has(const char *c) const {
    result_t result = get(c);
    return ((size_t) result.length == strlen(c));
  }

  template <class TM>
  bool trie_t<TM>::has(const char *c, const int size) const {
    OCCA_ERROR("Cannot search for a char* with size: " << size,
               0 < size);

    const bool usedAlloca = (size < 1024);
    char *c2 = usedAlloca ? ((char*) alloca(size + 1)) : (new char [size + 1]);
    c2[size] = '\0';
    ::memcpy(c2, c, size);

    const bool has_ = has(c2);
    if (!usedAlloca) {
      delete [] c2;
    }
    return has_;
  }

  template <class TM>
  void trie_t<TM>::print() {
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
