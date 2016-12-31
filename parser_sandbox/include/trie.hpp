#ifndef OCCA_TRIE_HEADER2
#define OCCA_TRIE_HEADER2

#include <iostream>
#include <vector>
#include <map>

namespace occa {
  class trieNode_t;
  typedef std::map<char, trieNode_t>    trieNodeMap_t;
  typedef trieNodeMap_t::iterator       trieNodeMapIterator;
  typedef trieNodeMap_t::const_iterator cTrieNodeMapIterator;

  //---[ Node ]-------------------------
  class trieNode_t {
  public:
    class result_t {
    public:
      int length;
      int valueIdx;

      result_t();
      result_t(const int length_, const int valueIdx);
    };

    int valueIdx;
    trieNodeMap_t leaves;

    trieNode_t();
    trieNode_t(const int valueIdx_);

    void add(const char *c, const int valueIdx_);

    int nodeCount() const;
    int getValueIdx(const char *c) const;
    result_t get(const char *c) const;
    result_t get(const char *c, const int length) const;
  };
  //====================================

  //---[ Trie ]-------------------------
  template <class TM>
  class trie_t {
  public:
    //---[ Result ]---------------------
    class result_t {
    public:
      int length;
      TM value;

      result_t();
      result_t(const int length_, const TM &value_ = TM());
      operator bool();
    };
    //==================================

    trieNode_t root;
    std::vector<TM> values;

    bool isFrozen;
    int nodeCount, baseNodeCount;
    char *chars;
    int *offsets, *leafCount;
    int *valueIndices;

    trie_t();

    void add(const char *c, const TM &value = TM());
    void add(const std::string &s, const TM &value = TM());

    void freeze();
    int freeze(const trieNode_t &node, int offset);
    void defrost();

    bool empty() const;

    result_t getFirst(const char *c) const;
    result_t trieGetFirst(const char *c) const;

    result_t get(const char *c) const;

    bool has(const char c) const;
    bool trieHas(const char c) const;

    bool has(const char *c) const;
    bool has(const char *c, const int size) const;

    void print();
  };
  //====================================
}

#include "trie.tpp"

#endif
