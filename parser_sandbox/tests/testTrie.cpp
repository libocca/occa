#include "occa/tools/testing.hpp"

#include "trie.hpp"

#include <time.h>
#include <stdlib.h>
#include "occa/tools/io.hpp"
#include "occa/tools/lex.hpp"

void testInsert(occa::trie<std::string> &trie);
void testSearch(occa::trie<std::string> &trie);
void testFrozenSearch(occa::trie<std::string> &trie);
void testRefreeze(occa::trie<std::string> &trie);

int main(const int argc, const char **argv) {
  occa::trie<std::string> trie;
  trie.autoFreeze = false;

  testInsert(trie);
  testSearch(trie);
  testFrozenSearch(trie);
  testRefreeze(trie);
}

void testInsert(occa::trie<std::string> &trie) {
  trie.add("blue"    , "blue");
  trie.add("blueblue", "blueblue");
  trie.add("boring"  , "boring");
  trie.add("glue"    , "glue");
  trie.add("good"    , "good");
}

void testSearch(occa::trie<std::string> &trie) {
  OCCA_TEST_COMPARE(true, trie.has("blue"));
  OCCA_TEST_COMPARE(true, trie.has("boring"));
  OCCA_TEST_COMPARE(true, trie.has("glue"));
  OCCA_TEST_COMPARE(true, trie.has("good"));

  OCCA_TEST_COMPARE(false, trie.has("red"));
  OCCA_TEST_COMPARE(false, trie.has("goo"));
  OCCA_TEST_COMPARE(false, trie.has("goods"));

  OCCA_TEST_COMPARE("blue"  , trie.getFirst("blue").value());
  OCCA_TEST_COMPARE("boring", trie.getFirst("boring").value());
  OCCA_TEST_COMPARE("glue"  , trie.getFirst("glue").value());
  OCCA_TEST_COMPARE("good"  , trie.getFirst("good").value());

  OCCA_TEST_COMPARE(""    , trie.getFirst("red").value());
  OCCA_TEST_COMPARE(""    , trie.getFirst("goo").value());
  OCCA_TEST_COMPARE("good", trie.getFirst("goods").value());

  OCCA_TEST_COMPARE("blue"  , trie.get("blue").value());
  OCCA_TEST_COMPARE("boring", trie.get("boring").value());
  OCCA_TEST_COMPARE("glue"  , trie.get("glue").value());
  OCCA_TEST_COMPARE("good"  , trie.get("good").value());

  OCCA_TEST_COMPARE("", trie.get("red").value());
  OCCA_TEST_COMPARE("", trie.get("goo").value());
  OCCA_TEST_COMPARE("", trie.get("goods").value());
}

void testFrozenSearch(occa::trie<std::string> &trie) {
  trie.freeze();
  OCCA_TEST_COMPARE(true, trie.has("blue"));
  OCCA_TEST_COMPARE(true, trie.has("boring"));
  OCCA_TEST_COMPARE(true, trie.has("glue"));
  OCCA_TEST_COMPARE(true, trie.has("good"));

  OCCA_TEST_COMPARE(false, trie.has("red"));
  OCCA_TEST_COMPARE(false, trie.has("goo"));
  OCCA_TEST_COMPARE(false, trie.has("goods"));

  OCCA_TEST_COMPARE("blue"  , trie.getFirst("blue").value());
  OCCA_TEST_COMPARE("boring", trie.getFirst("boring").value());
  OCCA_TEST_COMPARE("glue"  , trie.getFirst("glue").value());
  OCCA_TEST_COMPARE("good"  , trie.getFirst("good").value());

  OCCA_TEST_COMPARE(""    , trie.getFirst("red").value());
  OCCA_TEST_COMPARE(""    , trie.getFirst("goo").value());
  OCCA_TEST_COMPARE("good", trie.getFirst("goods").value());

  OCCA_TEST_COMPARE("blue"  , trie.get("blue").value());
  OCCA_TEST_COMPARE("boring", trie.get("boring").value());
  OCCA_TEST_COMPARE("glue"  , trie.get("glue").value());
  OCCA_TEST_COMPARE("good"  , trie.get("good").value());

  OCCA_TEST_COMPARE("", trie.get("red").value());
  OCCA_TEST_COMPARE("", trie.get("goo").value());
  OCCA_TEST_COMPARE("", trie.get("goods").value());
}

void testRefreeze(occa::trie<std::string> &trie) {
  OCCA_TEST_COMPARE(true , trie.isFrozen);
  OCCA_TEST_COMPARE(false, trie.has("red"));

  trie.add("red", "red");
  OCCA_TEST_COMPARE(false, trie.isFrozen);
  OCCA_TEST_COMPARE(true , trie.has("red"));

  OCCA_TEST_COMPARE("red", trie.get("red").value());

  trie.autoFreeze = true;
  trie.add("blue", "red");
  OCCA_TEST_COMPARE(true, trie.isFrozen);
  OCCA_TEST_COMPARE(true, trie.has("red"));

  OCCA_TEST_COMPARE("red", trie.get("red").value());
  OCCA_TEST_COMPARE("red", trie.get("blue").value());
}
