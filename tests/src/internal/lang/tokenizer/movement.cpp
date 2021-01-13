#include "utils.hpp"

void testSkipMethods();
void testPushPop();

using namespace occa::lang;

int main(const int argc, const char **argv) {
  testSkipMethods();
  testPushPop();

  return 0;
}

void testSkipMethods() {
  setStream("ab\nc\n\n\n\n\n\nd\ne");
  const char *c = source.c_str();

  tokenizer.skipTo('a');
  ASSERT_EQ('a', *tokenizer.fp.start);

  tokenizer.skipTo('b');
  ASSERT_EQ('b' , *tokenizer.fp.start);

  tokenizer.skipTo('e');
  ASSERT_EQ('e' , *tokenizer.fp.start);

  tokenizer.fp.start = c;
  tokenizer.skipTo("c\n");
  ASSERT_EQ(c + 2, tokenizer.fp.start);

  tokenizer.fp.start = c + 6;
  tokenizer.skipFrom("\n");
  ASSERT_EQ('d' , *tokenizer.fp.start);
}

void testPushPop() {
  setStream("a\nb\nc\nd\ne");
  const char *c = source.c_str();

  tokenizer.push();
  tokenizer.skipTo('c');
  ASSERT_EQ(3,
            tokenizer.fp.line);
  ASSERT_EQ(c + 4,
            tokenizer.fp.start);
  tokenizer.popAndRewind();
  ASSERT_EQ(1,
            tokenizer.fp.line);
  ASSERT_EQ(c + 0,
            tokenizer.fp.start);
  tokenizer.push();
  tokenizer.push();
  tokenizer.push();
  tokenizer.skipTo('c');
  tokenizer.pop();
  tokenizer.pop();
  tokenizer.pop();
  ASSERT_EQ(3,
            tokenizer.fp.line);
  ASSERT_EQ(c + 4,
            tokenizer.fp.start);
}
