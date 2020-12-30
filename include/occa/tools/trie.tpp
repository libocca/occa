#include <cstring>

#include <occa/defines.hpp>
#include <occa/internal/utils/sys.hpp>

#if (OCCA_OS & (OCCA_LINUX_OS | OCCA_MACOS_OS))
#  include <alloca.h>
#endif

namespace occa {
  //---[ Trie ]-------------------------
  //  ---[ Result ]---------------------
  template <class TM>
  trie<TM>::result_t::result_t() {}

  template <class TM>
  trie<TM>::result_t::result_t(const trie<TM> *trie__,
                               const int length_,
                               const int valueIndex_) :
    trie_(const_cast<trie<TM>*>(trie__)),
    length(length_),
    valueIndex(valueIndex_) {}

  template <class TM>
  bool trie<TM>::result_t::success() const {
    return (0 <= valueIndex);
  }

  template <class TM>
  const TM& trie<TM>::result_t::value() const {
    return ((0 <= valueIndex)
            ? trie_->values[valueIndex]
            : trie_->defaultValue);
  }

  template <class TM>
  TM& trie<TM>::result_t::value() {
    return ((0 <= valueIndex)
            ? trie_->values[valueIndex]
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
  trie<TM>::trie(const trie &other) {
    *this = other;
  }

  template <class TM>
  trie<TM>::~trie() {
    clear();
  }

  template <class TM>
  trie<TM>& trie<TM>::operator = (const trie &other) {
    if (this == &other) {
      return *this;
    }

    trie &other_ = const_cast<trie&>(other);

    other_.freeze();

    root         = other_.root;
    defaultValue = other_.defaultValue;
    values       = other_.values;

    isFrozen   = other_.isFrozen;
    autoFreeze = other_.autoFreeze;

    nodeCount     = other_.nodeCount;
    baseNodeCount = other_.baseNodeCount;

    chars        = new char[nodeCount + 1];
    offsets      = new int[nodeCount + 1];
    leafCount    = new int[nodeCount + 1];
    valueIndices = new int[nodeCount + 1];

    for (int i = 0; i < (nodeCount + 1); ++i) {
      chars[i]        = other_.chars[i];
      offsets[i]      = other_.offsets[i];
      leafCount[i]    = other_.leafCount[i];
      valueIndices[i] = other_.valueIndices[i];
    }

    return *this;
  }

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
  int trie<TM>::size() const {
    if (isFrozen) {
      return (int) values.size();
    }
    return root.size();
  }

  template <class TM>
  void trie<TM>::add(const char *c, const TM &value) {
    int valueIndex = root.getValueIndex(c);
    if (valueIndex < 0) {
      defrost();
      valueIndex = values.size();
      values.push_back(value);
      root.add(c, valueIndex);
      if (autoFreeze) {
        freeze();
      }
    } else {
      values[valueIndex] = value;
    }
  }

  template <class TM>
  void trie<TM>::add(const std::string &s, const TM &value) {
    add(s.c_str(), value);
  }

  template <class TM>
  void trie<TM>::remove(const char *c, const int length) {
    int valueIndex = root.getValueIndex(c);
    if (valueIndex >= 0) {
      const int length_ = ((length == INT_MAX)
                           ? strlen(c)
                           : length);
      defrost();
      root.remove(c, length_, valueIndex);
      if (autoFreeze) {
        freeze();
      }

      for (int i = (valueIndex + 1); i < (int) values.size(); ++i) {
        values[i - 1] = values[i];
      }
      values.pop_back();
    }
  }

  template <class TM>
  void trie<TM>::remove(const std::string &s) {
    remove(s.c_str(), (int) s.size());
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
      valueIndices[offset] = leafNode.valueIndex;

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
    int retValueIndex = -1;

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
        }
        else if (ci > cmid) {
          start = (OCCA_TRIE_MID + 1);
        }
        else {
          ++c;
          if (0 <= valueIndices[offset + OCCA_TRIE_MID]) {
            retLength   = (c - cStart);
            retValueIndex = valueIndices[offset + OCCA_TRIE_MID];
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
      return result_t(this, retLength, retValueIndex);
    }
    return result_t(this);
  }

  template <class TM>
  typename trie<TM>::result_t trie<TM>::trieGetLongest(const char *c, const int length) const {
    trieNode::result_t result = root.get(c, length);
    if (result.success()) {
      return result_t(this, result.length, result.valueIndex);
    }
    return result_t(this);
  }

  template <class TM>
  typename trie<TM>::result_t trie<TM>::get(const char *c, const int length) const {
    const int length_ = (length == INT_MAX) ? strlen(c) : length;
    trie<TM>::result_t result = getLongest(c, length_);
    if (result.length != length_) {
      result.length = 0;
      result.valueIndex = -1;
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
      "index      : ",
      "chars      : ",
      "offsets    : ",
      "leafCount  : ",
      "valueIndex : "
    };
    io::stdout << headers[0];
    for (int i = 0; i < nodeCount; ++i) {
      std::string spaces(i < 10 ? 2 : ((i < 100) ? 1 : 0), ' ');
      io::stdout << i << ' ' << spaces;
    }
    io::stdout << '\n' << headers[1];
    for (int i = 0; i < nodeCount; ++i) {
      io::stdout << chars[i] << "   ";
    }
    io::stdout << '\n' << headers[2];
    for (int i = 0; i < nodeCount; ++i) {
      const int offset = offsets[i];
      std::string spaces(offset < 10 ? 2 : ((offset < 100) ? 1 : 0), ' ');
      io::stdout << offsets[i] << ' ' << spaces;
    }
    io::stdout << '\n' << headers[3];
    for (int i = 0; i < nodeCount; ++i) {
      const int lcount = leafCount[i];
      std::string spaces(lcount < 10 ? 2 : ((lcount < 100) ? 1 : 0), ' ');
      io::stdout << leafCount[i] << ' ' << spaces;
    }
    io::stdout << '\n' << headers[4];
    for (int i = 0; i < nodeCount; ++i) {
      const int valueIndex = valueIndices[i];
      std::string spaces(valueIndex < 10 ? 2 : ((valueIndex < 100) ? 1 : 0), ' ');
      if (valueIndex < 0) {
        spaces = spaces.substr(1);
      }
      io::stdout << valueIndex << ' ' << spaces;
    }
    io::stdout << '\n';
    isFrozen = wasFrozen;
  }
  //====================================
}
