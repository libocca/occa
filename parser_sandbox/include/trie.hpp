#ifndef OCCA_TRIE_HEADER2
#define OCCA_TRIE_HEADER2

#include <iostream>
#include <map>

class trieNode_t;
typedef std::map<char, trieNode_t>    trieNodeMap_t;
typedef trieNodeMap_t::iterator       trieNodeMapIterator;
typedef trieNodeMap_t::const_iterator cTrieNodeMapIterator;

class trieNode_t {
public:
  bool isLeaf;
  trieNodeMap_t leaves;

  trieNode_t();

  void add(const char *c);

  int leafNodes() const;
  const char* get(const char *c) const;
};

class trie_t {
public:
  trieNode_t root;

  int nodeCount;
  int baseNodeCount;
  char *chars;
  int *offsets;
  int *leafCount;
  bool *isLeaf;

  trie_t();

  void add(const char *c);
  void add(const std::string &s);

  void freeze();
  int freeze(const trieNode_t &node, int offset);

  const char* get(const char *c) const;
  bool has(const char c) const;
  bool has(const char *c) const;

  void print() const;
};
#endif
