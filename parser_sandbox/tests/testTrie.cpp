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
  OCCA_TEST_COMPARE(trie.has("blue")   , true);
  OCCA_TEST_COMPARE(trie.has("boring") , true);
  OCCA_TEST_COMPARE(trie.has("glue")   , true);
  OCCA_TEST_COMPARE(trie.has("good")   , true);

  OCCA_TEST_COMPARE(trie.has("red")   , false);
  OCCA_TEST_COMPARE(trie.has("goo")   , false);
  OCCA_TEST_COMPARE(trie.has("goods") , false);

  OCCA_TEST_COMPARE(trie.getFirst("blue").value   , "blue");
  OCCA_TEST_COMPARE(trie.getFirst("boring").value , "boring");
  OCCA_TEST_COMPARE(trie.getFirst("glue").value   , "glue");
  OCCA_TEST_COMPARE(trie.getFirst("good").value   , "good");

  OCCA_TEST_COMPARE(trie.getFirst("red").value   , "");
  OCCA_TEST_COMPARE(trie.getFirst("goo").value   , "");
  OCCA_TEST_COMPARE(trie.getFirst("goods").value , "good");

  OCCA_TEST_COMPARE(trie.get("blue").value   , "blue");
  OCCA_TEST_COMPARE(trie.get("boring").value , "boring");
  OCCA_TEST_COMPARE(trie.get("glue").value   , "glue");
  OCCA_TEST_COMPARE(trie.get("good").value   , "good");

  OCCA_TEST_COMPARE(trie.get("red").value   , "");
  OCCA_TEST_COMPARE(trie.get("goo").value   , "");
  OCCA_TEST_COMPARE(trie.get("goods").value , "");
}

void testFrozenSearch(occa::trie_t<std::string> &trie) {
  trie.freeze();
  OCCA_TEST_COMPARE(trie.has("blue")   , true);
  OCCA_TEST_COMPARE(trie.has("boring") , true);
  OCCA_TEST_COMPARE(trie.has("glue")   , true);
  OCCA_TEST_COMPARE(trie.has("good")   , true);

  OCCA_TEST_COMPARE(trie.has("red")   , false);
  OCCA_TEST_COMPARE(trie.has("goo")   , false);
  OCCA_TEST_COMPARE(trie.has("goods") , false);

  OCCA_TEST_COMPARE(trie.getFirst("blue").value   , "blue");
  OCCA_TEST_COMPARE(trie.getFirst("boring").value , "boring");
  OCCA_TEST_COMPARE(trie.getFirst("glue").value   , "glue");
  OCCA_TEST_COMPARE(trie.getFirst("good").value   , "good");

  OCCA_TEST_COMPARE(trie.getFirst("red").value   , "");
  OCCA_TEST_COMPARE(trie.getFirst("goo").value   , "");
  OCCA_TEST_COMPARE(trie.getFirst("goods").value , "good");

  OCCA_TEST_COMPARE(trie.get("blue").value   , "blue");
  OCCA_TEST_COMPARE(trie.get("boring").value , "boring");
  OCCA_TEST_COMPARE(trie.get("glue").value   , "glue");
  OCCA_TEST_COMPARE(trie.get("good").value   , "good");

  OCCA_TEST_COMPARE(trie.get("red").value   , "");
  OCCA_TEST_COMPARE(trie.get("goo").value   , "");
  OCCA_TEST_COMPARE(trie.get("goods").value , "");
}

void testRefreeze(occa::trie_t<std::string> &trie) {
  OCCA_TEST_COMPARE(trie.isFrozen, true);
  OCCA_TEST_COMPARE(trie.has("red"), false);

  trie.add("red", "red");
  trie.add("blue", "red");
  OCCA_TEST_COMPARE(trie.isFrozen, false);
  OCCA_TEST_COMPARE(trie.has("red"), true);
  OCCA_TEST_COMPARE(trie.get("red").value, "red");
  OCCA_TEST_COMPARE(trie.get("blue").value, "red");

  trie.freeze();
  OCCA_TEST_COMPARE(trie.isFrozen, true);
  OCCA_TEST_COMPARE(trie.has("red"), true);
  OCCA_TEST_COMPARE(trie.get("red").value, "red");
  OCCA_TEST_COMPARE(trie.get("blue").value, "red");
}
