#include <cstring>

#include "occa/defines.hpp"
#include "occa/tools/sys.hpp"
#include "trie.hpp"

namespace occa {
  trieNode_t::trieNode_t() :
    isLeaf(false) {}

  void trieNode_t::add(const char *c) {
    if (*c == '\0') {
      isLeaf = true;
      return;
    }
    trieNode_t &newNode = leaves[*c];
    newNode.add(c + 1);
  }

  int trieNode_t::leafNodes() const {
    cTrieNodeMapIterator it = leaves.begin();
    int count = (int) leaves.size();
    while (it != leaves.end()) {
      count += it->second.leafNodes();
      ++it;
    }
    return count;
  }

  const char* trieNode_t::get(const char *c) const {
    cTrieNodeMapIterator it = leaves.find(*c);
    if (it != leaves.end()) {
      const char *ret = it->second.get(c + 1);
      return ((NULL == ret) && isLeaf) ? (c + 1) : ret;
    } else {
      return isLeaf ? c : NULL;
    }
  }

  trie_t::trie_t() :
    isFrozen(false),
    nodeCount(0),
    baseNodeCount(0),
    chars(NULL),
    offsets(NULL),
    leafCount(NULL),
    isLeaf(NULL) {}

  void trie_t::add(const char *c) {
    root.add(c);
    defrost();
  }

  void trie_t::add(const std::string &s) {
    root.add(s.c_str());
    defrost();
  }

  void trie_t::freeze() {
    if (isFrozen) {
      defrost();
    }

    nodeCount     = root.leafNodes();
    baseNodeCount = (int) root.leaves.size();

    chars     = new char[nodeCount + 1];
    offsets   = new int[nodeCount + 1];
    leafCount = new int[nodeCount + 1];
    isLeaf    = new bool[nodeCount + 1];

    chars[nodeCount]     = '\0';
    offsets[nodeCount]   = nodeCount;
    leafCount[nodeCount] = 0;
    isLeaf[nodeCount]    = false;

    freeze(root, 0);
    isFrozen = true;
  }

  int trie_t::freeze(const trieNode_t &node, int offset) {
    const trieNodeMap_t &leaves = node.leaves;
    cTrieNodeMapIterator leaf = leaves.begin();
    int leafOffset = offset + (int) leaves.size();

    while (leaf != leaves.end()) {
      const trieNode_t &leafNode = leaf->second;
      chars[offset]     = leaf->first;
      offsets[offset]   = leafOffset;
      leafCount[offset] = (int) leafNode.leaves.size();
      isLeaf[offset]    = leafNode.isLeaf;

      leafOffset = freeze(leafNode, leafOffset);
      ++offset;
      ++leaf;
    }
    return leafOffset;
  }

  void trie_t::defrost() {
    if (isFrozen) {
      delete [] chars;
      delete [] offsets;
      delete [] leafCount;
      delete [] isLeaf;

      chars     = NULL;
      offsets   = NULL;
      leafCount = NULL;
      isLeaf    = NULL;

      isFrozen = false;
    }
  }

  const char* trie_t::get(const char *c) const {
    if (!isFrozen) {
      return trieGet(c);
    }
    const char *cStart = c;
    const char *ret    = c;

    int offset = 0;
    int count = baseNodeCount;
    while (count) {
      const char ci = *c;
      bool found = false;
      for (int i = 0; i < count; ++i) {
        if (ci == chars[offset + i]) {
          ++c;
          if (isLeaf[offset]) {
            ret = c;
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
    // Move to the first char not in the trie
    return (ret > cStart) ? ret : cStart;
  }

  const char* trie_t::trieGet(const char *c) const {
    const char *found = root.get(c);
    return found ? found : c;
  }

  bool trie_t::has(const char c) const {
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

  bool trie_t::trieHas(const char c) const {
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

  bool trie_t::has(const char *c) const {
    const char *cEnd = get(c);
    return ((size_t) (cEnd - c) == strlen(c));
  }

  bool trie_t::has(const char *c, const int size) const {
    OCCA_ERROR("Cannot search for a char* with size: " << size,
               0 < size);

    char *c2 = new char[size + 1];
    c2[size] = '\0';
    ::memcpy(c2, c, size);

    const bool has_ = has(c2);
    delete [] c2;
    return has_;
  }

  void trie_t::print() {
    const bool wasFrozen = isFrozen;
    if (!isFrozen) {
      freeze();
    }
    const std::string headers[] = {
      "index    : ",
      "chars    : ",
      "offsets  : ",
      "leafCount: ",
      "isLeaf   : "
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
      std::cout << isLeaf[i] << "   ";
    }
    std::cout << '\n';
    isFrozen = wasFrozen;
  }
}
