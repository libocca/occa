#ifndef OCCA_INTERNAL_UTILS_TRIE_HEADER
#define OCCA_INTERNAL_UTILS_TRIE_HEADER

#include <iostream>
#include <vector>
#include <map>

#include <limits.h>

namespace occa {
  class trieNode;
  typedef std::map<char, trieNode>      trieNodeMap_t;
  typedef trieNodeMap_t::iterator       trieNodeMapIterator;
  typedef trieNodeMap_t::const_iterator cTrieNodeMapIterator;

  //---[ Node ]-------------------------
  class trieNode {
  public:
    class result_t {
    public:
      trieNode *node;
      int length;
      int valueIndex;

      result_t();
      result_t(trieNode *node_,
               const int length_,
               const int valueIndex);

      inline bool success() const {
        return (0 <= valueIndex);
      }
    };

    int valueIndex;
    trieNodeMap_t leaves;

    trieNode();
    trieNode(const int valueIndex_);

    int size() const;
    int nodeCount() const;

    int getValueIndex(const char *c) const;

    void add(const char *c, const int valueIndex_);

    result_t get(const char *c,
                 const int length) const;
    result_t get(const char *c,
                 const int cIndex,
                 const int length) const;

    void remove(const char *c,
                const int valueIndex_);
    void remove(const char *c,
                const int length,
                const int valueIndex_);
    bool nestedRemove(const char *c,
                      const int length,
                      const int valueIndex_);

    void decrementIndex(const int valueIndex_);
  };
  //====================================

  //---[ Trie ]-------------------------
  template <class TM>
  class trie {
  public:
    //---[ Result ]---------------------
    class result_t {
    public:
      trie<TM> *trie_;
      int length;
      int valueIndex;

      result_t();
      result_t(const trie<TM> *trie__,
               const int length_ = 0,
               const int valueIndex_ = -1);

      inline bool success() const;
      inline const TM& value() const;
      inline TM& value();
    };
    //==================================

    trieNode root;
    TM defaultValue;
    std::vector<TM> values;

    bool isFrozen, autoFreeze;
    int nodeCount, baseNodeCount;
    char *chars;
    int *offsets, *leafCount;
    int *valueIndices;

    trie();
    trie(const trie &other);
    ~trie();

    trie& operator = (const trie &other);

    void clear();
    bool isEmpty() const;
    int size() const;

    void add(const char *c, const TM &value = TM());
    void add(const std::string &s, const TM &value = TM());

    void remove(const char *c,
                const int length = INT_MAX);

    void remove(const std::string &s);

    void freeze();
    int freeze(const trieNode &node, int offset);
    void defrost();

    result_t getLongest(const char *c,
                        const int length = INT_MAX) const;

    result_t trieGetLongest(const char *c,
                            const int length) const;

    inline result_t getLongest(const std::string &s) const {
      return getLongest(s.c_str(), (int) s.size());
    }

    result_t get(const char *c,
                 const int length = INT_MAX) const;

    inline result_t get(const std::string &s) const {
      return get(s.c_str(), (int) s.size());
    }

    bool has(const char c) const;
    bool trieHas(const char c) const;

    bool has(const char *c) const;
    bool has(const char *c,
             const int size) const;

    inline bool has(const std::string &s) const {
      return has(s.c_str(), s.size());
    }

    void print();
  };
  //====================================
}

#include "trie.tpp"

#endif
