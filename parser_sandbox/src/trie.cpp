#include "trie.hpp"

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
    return it->second.get(c + 1);
  } else {
    return isLeaf ? c : NULL;
  }
}

trie_t::trie_t() :
  nodeCount(0),
  baseNodeCount(0) {}

void trie_t::add(const char *c) {
  root.add(c);
}

void trie_t::add(const std::string &s) {
  root.add(s.c_str());
}

void trie_t::freeze() {
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

const char* trie_t::get(const char *c) const {
  const char *ret = c;

  int offset = 0;
  int count = baseNodeCount;
  while (count) {
    const char ci = *c;
    int i = 0;
    for (; i < count; ++i) {
      if (ci == chars[offset + i]) {
        offset = offsets[offset + i];
        count  = leafCount[offset];

        ++c;
        if (isLeaf[offset]) {
          ret = c;
        }

        break;
      }
    }
    if (i == count) {
      break;
    }
  }
  return ret;
}

bool trie_t::has(const char c) const {
  for (int i = 0; i < baseNodeCount; ++i) {
    if (chars[i] == c) {
      return true;
    }
  }
  return false;
}

bool trie_t::has(const char *c) const {
  const char *cEnd = get(c);
  return (c != cEnd);
}

void trie_t::print() const {
  const std::string headers[] = {
    "index    : ",
    "chars    : ",
    "offsets  : ",
    "leafCount: ",
    "isLeaf   : "
  };
  std::cout << headers[0];
  for (int i = 0; i < nodeCount; ++i) {
    std::cout << i << ' ' << std::string(2 - (i < 10 ? 0 : 1), ' ');
  }
  std::cout << '\n' << headers[1];
  for (int i = 0; i < nodeCount; ++i) {
    std::cout << chars[i] << "   ";
  }
  std::cout << '\n' << headers[2];
  for (int i = 0; i < nodeCount; ++i) {
    std::cout << offsets[i] << ' ' << std::string(2 - (offsets[i] < 10 ? 0 : 1), ' ');
  }
  std::cout << '\n' << headers[3];
  for (int i = 0; i < nodeCount; ++i) {
    std::cout << leafCount[i] << ' ' << std::string(2 - (leafCount[i] < 10 ? 0 : 1), ' ');
  }
  std::cout << '\n' << headers[4];
  for (int i = 0; i < nodeCount; ++i) {
    std::cout << isLeaf[i] << "   ";
  }
  std::cout << '\n';
}
