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
#include <time.h>
#include <stdlib.h>

#include <occa/tools/testing.hpp>
#include <occa/tools/trie.hpp>
#include <occa/io.hpp>
#include <occa/tools/lex.hpp>

void testInsert(occa::trie<std::string> &trie);
void testSearch(occa::trie<std::string> &trie);
void testFrozenSearch(occa::trie<std::string> &trie);
void testRemoval(occa::trie<std::string> &trie);
void testRefreeze(occa::trie<std::string> &trie);

int main(const int argc, const char **argv) {
  occa::trie<std::string> trie;
  trie.autoFreeze = false;

  testInsert(trie);
  testSearch(trie);
  testFrozenSearch(trie);
  testRemoval(trie);
  testRefreeze(trie);

  return 0;
}

void testInsert(occa::trie<std::string> &trie) {
  trie.add("blue"    , "blue");
  trie.add("blueblue", "blueblue");
  trie.add("boring"  , "boring");
  trie.add("glue"    , "glue");
  trie.add("good"    , "good");

  ASSERT_EQ(5, trie.size());
}

void testSearch(occa::trie<std::string> &trie) {
  ASSERT_TRUE(trie.has("blue"));
  ASSERT_TRUE(trie.has("boring"));
  ASSERT_TRUE(trie.has("glue"));
  ASSERT_TRUE(trie.has("good"));

  ASSERT_FALSE(trie.has("red"));
  ASSERT_FALSE(trie.has("goo"));
  ASSERT_FALSE(trie.has("goods"));

  ASSERT_EQ("blue"  , trie.getLongest("blue").value());
  ASSERT_EQ("boring", trie.getLongest("boring").value());
  ASSERT_EQ("glue"  , trie.getLongest("glue").value());
  ASSERT_EQ("good"  , trie.getLongest("good").value());

  ASSERT_EQ("blueblue", trie.getLongest("blueblue").value());
  ASSERT_EQ(""        , trie.getLongest("red").value());
  ASSERT_EQ(""        , trie.getLongest("goo").value());
  ASSERT_EQ("good"    , trie.getLongest("goods").value());

  ASSERT_EQ("blue"  , trie.get("blue").value());
  ASSERT_EQ("boring", trie.get("boring").value());
  ASSERT_EQ("glue"  , trie.get("glue").value());
  ASSERT_EQ("good"  , trie.get("good").value());

  ASSERT_EQ("blueblue", trie.get("blueblue").value());
  ASSERT_EQ(""        , trie.get("red").value());
  ASSERT_EQ(""        , trie.get("goo").value());
  ASSERT_EQ(""        , trie.get("goods").value());
}

void testFrozenSearch(occa::trie<std::string> &trie) {
  trie.freeze();
  ASSERT_TRUE(trie.has("blue"));
  ASSERT_TRUE(trie.has("boring"));
  ASSERT_TRUE(trie.has("glue"));
  ASSERT_TRUE(trie.has("good"));

  ASSERT_FALSE(trie.has("red"));
  ASSERT_FALSE(trie.has("goo"));
  ASSERT_FALSE(trie.has("goods"));

  ASSERT_EQ("blue"  , trie.getLongest("blue").value());
  ASSERT_EQ("boring", trie.getLongest("boring").value());
  ASSERT_EQ("glue"  , trie.getLongest("glue").value());
  ASSERT_EQ("good"  , trie.getLongest("good").value());

  ASSERT_EQ("blueblue", trie.getLongest("blueblue").value());
  ASSERT_EQ(""        , trie.getLongest("red").value());
  ASSERT_EQ(""        , trie.getLongest("goo").value());
  ASSERT_EQ("good"    , trie.getLongest("goods").value());

  ASSERT_EQ("blue"  , trie.get("blue").value());
  ASSERT_EQ("boring", trie.get("boring").value());
  ASSERT_EQ("glue"  , trie.get("glue").value());
  ASSERT_EQ("good"  , trie.get("good").value());

  ASSERT_EQ("blueblue", trie.get("blueblue").value());
  ASSERT_EQ(""        , trie.get("red").value());
  ASSERT_EQ(""        , trie.get("goo").value());
  ASSERT_EQ(""        , trie.get("goods").value());
}

void testRemoval(occa::trie<std::string> &trie) {
  ASSERT_EQ(5, trie.size());

  ASSERT_TRUE(trie.has("blue"));
  trie.remove("blue");
  ASSERT_FALSE(trie.has("blue"));

  ASSERT_TRUE(trie.has("blueblue"));
  trie.remove("blueblue");
  ASSERT_FALSE(trie.has("blueblue"));

  ASSERT_TRUE(trie.has("boring"));
  trie.remove("boring");
  ASSERT_FALSE(trie.has("boring"));

  ASSERT_TRUE(trie.has("glue"));
  trie.remove("glue");
  ASSERT_FALSE(trie.has("glue"));

  ASSERT_TRUE(trie.has("good"));
  trie.remove("good");
  ASSERT_FALSE(trie.has("good"));

  ASSERT_EQ(0, trie.size());
}

void testRefreeze(occa::trie<std::string> &trie) {
  trie.clear();
  ASSERT_EQ(0, trie.size());

  trie.freeze();
  ASSERT_TRUE(trie.isFrozen);
  ASSERT_FALSE(trie.has("red"));

  trie.add("red", "red");
  ASSERT_FALSE(trie.isFrozen);
  ASSERT_TRUE(trie.has("red"));

  ASSERT_EQ("red", trie.get("red").value());

  trie.autoFreeze = true;
  trie.add("blue", "red");
  ASSERT_TRUE(trie.isFrozen);
  ASSERT_TRUE(trie.has("blue"));

  ASSERT_EQ("red", trie.get("red").value());
  ASSERT_EQ("red", trie.get("blue").value());

  ASSERT_EQ(2, trie.size());

  trie.remove("red");
  trie.remove("blue");
  ASSERT_EQ(0, trie.size());
  ASSERT_TRUE(trie.isFrozen);
}
