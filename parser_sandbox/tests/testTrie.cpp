/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 David Medina and Tim Warburton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */
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
  OCCA_ASSERT_TRUE(trie.has("blue"));
  OCCA_ASSERT_TRUE(trie.has("boring"));
  OCCA_ASSERT_TRUE(trie.has("glue"));
  OCCA_ASSERT_TRUE(trie.has("good"));

  OCCA_ASSERT_FALSE(trie.has("red"));
  OCCA_ASSERT_FALSE(trie.has("goo"));
  OCCA_ASSERT_FALSE(trie.has("goods"));

  OCCA_ASSERT_EQUAL("blue"  , trie.getLongest("blue").value());
  OCCA_ASSERT_EQUAL("boring", trie.getLongest("boring").value());
  OCCA_ASSERT_EQUAL("glue"  , trie.getLongest("glue").value());
  OCCA_ASSERT_EQUAL("good"  , trie.getLongest("good").value());

  OCCA_ASSERT_EQUAL("blueblue", trie.getLongest("blueblue").value());
  OCCA_ASSERT_EQUAL(""        , trie.getLongest("red").value());
  OCCA_ASSERT_EQUAL(""        , trie.getLongest("goo").value());
  OCCA_ASSERT_EQUAL("good"    , trie.getLongest("goods").value());

  OCCA_ASSERT_EQUAL("blue"  , trie.get("blue").value());
  OCCA_ASSERT_EQUAL("boring", trie.get("boring").value());
  OCCA_ASSERT_EQUAL("glue"  , trie.get("glue").value());
  OCCA_ASSERT_EQUAL("good"  , trie.get("good").value());

  OCCA_ASSERT_EQUAL("blueblue", trie.get("blueblue").value());
  OCCA_ASSERT_EQUAL(""        , trie.get("red").value());
  OCCA_ASSERT_EQUAL(""        , trie.get("goo").value());
  OCCA_ASSERT_EQUAL(""        , trie.get("goods").value());
}

void testFrozenSearch(occa::trie<std::string> &trie) {
  trie.freeze();
  OCCA_ASSERT_TRUE(trie.has("blue"));
  OCCA_ASSERT_TRUE(trie.has("boring"));
  OCCA_ASSERT_TRUE(trie.has("glue"));
  OCCA_ASSERT_TRUE(trie.has("good"));

  OCCA_ASSERT_FALSE(trie.has("red"));
  OCCA_ASSERT_FALSE(trie.has("goo"));
  OCCA_ASSERT_FALSE(trie.has("goods"));

  OCCA_ASSERT_EQUAL("blue"  , trie.getLongest("blue").value());
  OCCA_ASSERT_EQUAL("boring", trie.getLongest("boring").value());
  OCCA_ASSERT_EQUAL("glue"  , trie.getLongest("glue").value());
  OCCA_ASSERT_EQUAL("good"  , trie.getLongest("good").value());

  OCCA_ASSERT_EQUAL("blueblue", trie.getLongest("blueblue").value());
  OCCA_ASSERT_EQUAL(""        , trie.getLongest("red").value());
  OCCA_ASSERT_EQUAL(""        , trie.getLongest("goo").value());
  OCCA_ASSERT_EQUAL("good"    , trie.getLongest("goods").value());

  OCCA_ASSERT_EQUAL("blue"  , trie.get("blue").value());
  OCCA_ASSERT_EQUAL("boring", trie.get("boring").value());
  OCCA_ASSERT_EQUAL("glue"  , trie.get("glue").value());
  OCCA_ASSERT_EQUAL("good"  , trie.get("good").value());

  OCCA_ASSERT_EQUAL("blueblue", trie.get("blueblue").value());
  OCCA_ASSERT_EQUAL(""        , trie.get("red").value());
  OCCA_ASSERT_EQUAL(""        , trie.get("goo").value());
  OCCA_ASSERT_EQUAL(""        , trie.get("goods").value());
}

void testRefreeze(occa::trie<std::string> &trie) {
  OCCA_ASSERT_TRUE(trie.isFrozen);
  OCCA_ASSERT_FALSE(trie.has("red"));

  trie.add("red", "red");
  OCCA_ASSERT_FALSE(trie.isFrozen);
  OCCA_ASSERT_TRUE(trie.has("red"));

  OCCA_ASSERT_EQUAL("red", trie.get("red").value());

  trie.autoFreeze = true;
  trie.add("blue", "red");
  OCCA_ASSERT_TRUE(trie.isFrozen);
  OCCA_ASSERT_TRUE(trie.has("red"));

  OCCA_ASSERT_EQUAL("red", trie.get("red").value());
  OCCA_ASSERT_EQUAL("red", trie.get("blue").value());
}
