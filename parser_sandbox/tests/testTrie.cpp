#include "occa/defines.hpp"
#include "occa/tools/sys.hpp"
#include "occa/tools/testing.hpp"

#include "trie.hpp"

void testInsert(occa::trie_t<std::string> &trie);
void testSearch(occa::trie_t<std::string> &trie);
void testFrozenSearch(occa::trie_t<std::string> &trie);
void testRefreeze(occa::trie_t<std::string> &trie);

int main(const int argc, const char **argv) {
  occa::trie_t<std::string> trie;
  testInsert(trie);
  testSearch(trie);
  testFrozenSearch(trie);
  testRefreeze(trie);
}

void testInsert(occa::trie_t<std::string> &trie) {
  trie.add("blue"    , "blue");
  trie.add("blueblue", "blueblue");
  trie.add("boring"  , "boring");
  trie.add("glue"    , "glue");
  trie.add("good"    , "good");
}

void testSearch(occa::trie_t<std::string> &trie) {
  occa::testing::compare(trie.has("blue")   , true);
  occa::testing::compare(trie.has("boring") , true);
  occa::testing::compare(trie.has("glue")   , true);
  occa::testing::compare(trie.has("good")   , true);

  occa::testing::compare(trie.has("red")   , false);
  occa::testing::compare(trie.has("goo")   , false);
  occa::testing::compare(trie.has("goods") , false);

  occa::testing::compare(trie.getFirst("blue").value   , "blue");
  occa::testing::compare(trie.getFirst("boring").value , "boring");
  occa::testing::compare(trie.getFirst("glue").value   , "glue");
  occa::testing::compare(trie.getFirst("good").value   , "good");

  occa::testing::compare(trie.getFirst("red").value   , "");
  occa::testing::compare(trie.getFirst("goo").value   , "");
  occa::testing::compare(trie.getFirst("goods").value , "good");

  occa::testing::compare(trie.get("blue").value   , "blue");
  occa::testing::compare(trie.get("boring").value , "boring");
  occa::testing::compare(trie.get("glue").value   , "glue");
  occa::testing::compare(trie.get("good").value   , "good");

  occa::testing::compare(trie.get("red").value   , "");
  occa::testing::compare(trie.get("goo").value   , "");
  occa::testing::compare(trie.get("goods").value , "");
}

void testFrozenSearch(occa::trie_t<std::string> &trie) {
  trie.freeze();
  occa::testing::compare(trie.has("blue")   , true);
  occa::testing::compare(trie.has("boring") , true);
  occa::testing::compare(trie.has("glue")   , true);
  occa::testing::compare(trie.has("good")   , true);

  occa::testing::compare(trie.has("red")   , false);
  occa::testing::compare(trie.has("goo")   , false);
  occa::testing::compare(trie.has("goods") , false);

  occa::testing::compare(trie.getFirst("blue").value   , "blue");
  occa::testing::compare(trie.getFirst("boring").value , "boring");
  occa::testing::compare(trie.getFirst("glue").value   , "glue");
  occa::testing::compare(trie.getFirst("good").value   , "good");

  occa::testing::compare(trie.getFirst("red").value   , "");
  occa::testing::compare(trie.getFirst("goo").value   , "");
  occa::testing::compare(trie.getFirst("goods").value , "good");

  occa::testing::compare(trie.get("blue").value   , "blue");
  occa::testing::compare(trie.get("boring").value , "boring");
  occa::testing::compare(trie.get("glue").value   , "glue");
  occa::testing::compare(trie.get("good").value   , "good");

  occa::testing::compare(trie.get("red").value   , "");
  occa::testing::compare(trie.get("goo").value   , "");
  occa::testing::compare(trie.get("goods").value , "");
}

void testRefreeze(occa::trie_t<std::string> &trie) {
  occa::testing::compare(trie.isFrozen, true);
  occa::testing::compare(trie.has("red"), false);

  trie.add("red", "red");
  trie.add("blue", "red");
  occa::testing::compare(trie.isFrozen, false);
  occa::testing::compare(trie.has("red"), true);
  occa::testing::compare(trie.get("red").value, "red");
  occa::testing::compare(trie.get("blue").value, "red");

  trie.freeze();
  occa::testing::compare(trie.isFrozen, true);
  occa::testing::compare(trie.has("red"), true);
  occa::testing::compare(trie.get("red").value, "red");
  occa::testing::compare(trie.get("blue").value, "red");
}
