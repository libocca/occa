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
#include <sstream>

#include <occa/tools/testing.hpp>

#include <occa/lang/exprNode.hpp>
#include <occa/lang/token.hpp>
#include <occa/lang/tokenizer.hpp>
#include <occa/lang/tokenContext.hpp>

using namespace occa::lang;

void testMethods();
void testPairs();

std::string source;

void setupContext(tokenContext &context,
                  const std::string &source_) {
  source = source_;
  context.clear();
  context.tokens = tokenizer_t::tokenize(source);
  context.setup();
}

int main(const int argc, const char **argv) {
  testMethods();
  testPairs();

  return 0;
}

void testMethods() {
  tokenContext context;
  ASSERT_EQ(0, context.tp.start);
  ASSERT_EQ(0, context.tp.end);

  newlineToken    *newline    = new newlineToken(originSource::string);
  identifierToken *identifier = new identifierToken(originSource::string,
                                                    "identifier");
  primitiveToken  *primitive  = new primitiveToken(originSource::string,
                                                   1, "1");

  context.tokens.push_back(newline);
  context.tokens.push_back(identifier);
  context.tokens.push_back(primitive);

  ASSERT_EQ(0, context.tp.start);
  ASSERT_EQ(0, context.tp.end);

  context.setup();
  ASSERT_EQ(0, context.tp.start);
  ASSERT_EQ(3, context.tp.end);

  ASSERT_EQ(3, context.size());
  ASSERT_EQ((token_t*) newline,
            context[0]);
  ASSERT_EQ((token_t*) identifier,
            context[1]);
  ASSERT_EQ((token_t*) primitive,
            context[2]);

  // Out-of-bounds
  ASSERT_EQ((token_t*) NULL,
            context[-1]);
  ASSERT_EQ((token_t*) NULL,
            context[3]);

  context.push(1, 2);
  ASSERT_EQ(1, context.tp.start);
  ASSERT_EQ(2, context.tp.end);

  ASSERT_EQ((token_t*) identifier,
            context[0]);
  ASSERT_EQ((token_t*) NULL,
            context[1]);

  tokenRange prev = context.pop();
  ASSERT_EQ(0, context.tp.start);
  ASSERT_EQ(3, context.tp.end);
  ASSERT_EQ(1, prev.start);
  ASSERT_EQ(2, prev.end);

  context.set(1);
  ASSERT_EQ(1, context.tp.start);
  ASSERT_EQ(3, context.tp.end);

  context.push();
  context.set(1, 2);
  ASSERT_EQ(2, context.tp.start);
  ASSERT_EQ(3, context.tp.end);

  prev = context.pop();
  ASSERT_EQ(1, context.tp.start);
  ASSERT_EQ(3, context.tp.end);
  ASSERT_EQ(1, prev.start);
  ASSERT_EQ(2, prev.end);
}

void testPairs() {
  tokenContext context;
  // 0  | [<<<] [(]
  // 2  |   [[]
  // 3  |     [{] [1] [}] [,] [{] [2] [}]
  // 10 |   []] [,] [[]
  // 13 |     [{] [3] [}] [,] [{] [4] [}]
  // 20 |   []]
  // 21 | [)] [>>>]
  setupContext(context, "<<<([{1},{2}], [{3},{4}])>>>");
  ASSERT_EQ(8, (int) context.pairs.size());
  ASSERT_EQ(22, context.pairs[0]);  // <<<
  ASSERT_EQ(21, context.pairs[1]);  //  (
  ASSERT_EQ(10, context.pairs[2]);  //   [
  ASSERT_EQ(5 , context.pairs[3]);  //    {
  ASSERT_EQ(9 , context.pairs[7]);  //    {
  ASSERT_EQ(20, context.pairs[12]); //   [
  ASSERT_EQ(15, context.pairs[13]); //    {
  ASSERT_EQ(19, context.pairs[17]); //    {

  // Test pair range pushes
  intIntMap::iterator it = context.pairs.begin();
  while (it != context.pairs.end()) {
    const int pairStart = it->first;
    const int pairEnd   = it->second;
    context.pushPairRange(pairStart);
    ASSERT_EQ(pairEnd - pairStart - 1,
              context.size());
    context.pop();
    ++it;
  }

  // Test pair range pop
  // [{1}, {2}]
  context.pushPairRange(2);
  // {1}
  context.pushPairRange(0);
  // ,
  context.popAndSkip();
  ASSERT_EQ_BINARY(tokenType::op,
                   context[0]->type());
  ASSERT_EQ(operatorType::comma,
            context[0]->to<operatorToken>().opType());
  // {2}
  context.pushPairRange(1);
  context.popAndSkip();
  ASSERT_EQ(context.tp.start,
            context.tp.end);


  std::cerr << "Testing pair errors:\n";

  setupContext(context, "1, 2)");
  setupContext(context, "1, 2]");
  setupContext(context, "1, 2}");
  setupContext(context, "1, 2>>>");


  setupContext(context, "[1, 2)");
  setupContext(context, "{1, 2]");
  setupContext(context, "<<<1, 2}");
  setupContext(context, "(1, 2>>>");
}
