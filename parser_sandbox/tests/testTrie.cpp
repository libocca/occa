#include "occa/defines.hpp"
#include "occa/tools/sys.hpp"
#include "occa/tools/testing.hpp"

#include "trie.hpp"

void testInsert(occa::trie_t &trie);
void testSearch(occa::trie_t &trie);
void testFrozenSearch(occa::trie_t &trie);
void testRefreeze(occa::trie_t &trie);

int main(const int argc, const char **argv) {
  occa::trie_t trie;
  testInsert(trie);
  testSearch(trie);
  testFrozenSearch(trie);
  testRefreeze(trie);
}

void testInsert(occa::trie_t &trie) {
  trie.add("blue");
  trie.add("blueblue");
  trie.add("boring");
  trie.add("glue");
  trie.add("good");
}

void testSearch(occa::trie_t &trie) {
  occa::testing::compare(trie.has("blue")   , true);
  occa::testing::compare(trie.has("boring") , true);
  occa::testing::compare(trie.has("glue")   , true);
  occa::testing::compare(trie.has("good")   , true);

  occa::testing::compare(trie.has("red")   , false);
  occa::testing::compare(trie.has("goo")   , false);
  occa::testing::compare(trie.has("goods") , false);
}

void testFrozenSearch(occa::trie_t &trie) {
  trie.freeze();
  occa::testing::compare(trie.has("blue")   , true);
  occa::testing::compare(trie.has("boring") , true);
  occa::testing::compare(trie.has("glue")   , true);
  occa::testing::compare(trie.has("good")   , true);

  occa::testing::compare(trie.has("red")   , false);
  occa::testing::compare(trie.has("goo")   , false);
  occa::testing::compare(trie.has("goods") , false);
}

void testRefreeze(occa::trie_t &trie) {
  occa::testing::compare(trie.isFrozen, true);
  occa::testing::compare(trie.has("red"), false);

  trie.add("red");
  occa::testing::compare(trie.isFrozen, false);
  occa::testing::compare(trie.has("red"), true);

  trie.freeze();
  occa::testing::compare(trie.isFrozen, true);
  occa::testing::compare(trie.has("red"), true);
}
