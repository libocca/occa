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

#include "occa/tools/testing.hpp"

#include "exprNode.hpp"
#include "token.hpp"
#include "tokenizer.hpp"
#include "tokenContext.hpp"

using namespace occa::lang;

void testMethods();
void testPairs();
void testExpression();

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
  testExpression();

  return 0;
}

void testMethods() {
  tokenContext context;
  OCCA_ASSERT_EQUAL(0, context.tp.start);
  OCCA_ASSERT_EQUAL(0, context.tp.end);

  newlineToken    *newline    = new newlineToken(originSource::string);
  identifierToken *identifier = new identifierToken(originSource::string,
                                                    "identifier");
  primitiveToken  *primitive  = new primitiveToken(originSource::string,
                                                   1, "1");

  context.tokens.push_back(newline);
  context.tokens.push_back(identifier);
  context.tokens.push_back(primitive);

  OCCA_ASSERT_EQUAL(0, context.tp.start);
  OCCA_ASSERT_EQUAL(0, context.tp.end);

  context.setup();
  OCCA_ASSERT_EQUAL(0, context.tp.start);
  OCCA_ASSERT_EQUAL(3, context.tp.end);

  OCCA_ASSERT_EQUAL(3, context.size());
  OCCA_ASSERT_EQUAL((token_t*) newline,
                    context[0]);
  OCCA_ASSERT_EQUAL((token_t*) identifier,
                    context[1]);
  OCCA_ASSERT_EQUAL((token_t*) primitive,
                    context[2]);

  // Out-of-bounds
  OCCA_ASSERT_EQUAL((token_t*) NULL,
                    context[-1]);
  OCCA_ASSERT_EQUAL((token_t*) NULL,
                    context[3]);

  context.push(1, 2);
  OCCA_ASSERT_EQUAL(1, context.tp.start);
  OCCA_ASSERT_EQUAL(2, context.tp.end);

  OCCA_ASSERT_EQUAL((token_t*) identifier,
                    context[0]);
  OCCA_ASSERT_EQUAL((token_t*) NULL,
                    context[1]);

  tokenRange prev = context.pop();
  OCCA_ASSERT_EQUAL(0, context.tp.start);
  OCCA_ASSERT_EQUAL(3, context.tp.end);
  OCCA_ASSERT_EQUAL(1, prev.start);
  OCCA_ASSERT_EQUAL(2, prev.end);

  context.set(1);
  OCCA_ASSERT_EQUAL(1, context.tp.start);
  OCCA_ASSERT_EQUAL(3, context.tp.end);

  context.push();
  context.set(1, 2);
  OCCA_ASSERT_EQUAL(2, context.tp.start);
  OCCA_ASSERT_EQUAL(3, context.tp.end);

  prev = context.pop();
  OCCA_ASSERT_EQUAL(1, context.tp.start);
  OCCA_ASSERT_EQUAL(3, context.tp.end);
  OCCA_ASSERT_EQUAL(1, prev.start);
  OCCA_ASSERT_EQUAL(2, prev.end);
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
  OCCA_ASSERT_EQUAL(8, (int) context.pairs.size());
  OCCA_ASSERT_EQUAL(22, context.pairs[0]);  // <<<
  OCCA_ASSERT_EQUAL(21, context.pairs[1]);  //  (
  OCCA_ASSERT_EQUAL(10, context.pairs[2]);  //   [
  OCCA_ASSERT_EQUAL(5 , context.pairs[3]);  //    {
  OCCA_ASSERT_EQUAL(9 , context.pairs[7]);  //    {
  OCCA_ASSERT_EQUAL(20, context.pairs[12]); //   [
  OCCA_ASSERT_EQUAL(15, context.pairs[13]); //    {
  OCCA_ASSERT_EQUAL(19, context.pairs[17]); //    {

  // Test pair range pushes
  intIntMap::iterator it = context.pairs.begin();
  while (it != context.pairs.end()) {
    const int pairStart = it->first;
    const int pairEnd   = it->second;
    context.pushPairRange(pairStart);
    OCCA_ASSERT_EQUAL(pairEnd - pairStart - 1,
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
  OCCA_ASSERT_EQUAL_BINARY(tokenType::op,
                           context[0]->type());
  OCCA_ASSERT_EQUAL(operatorType::comma,
                    context[0]->to<operatorToken>().opType());
  // {2}
  context.pushPairRange(1);
  context.popAndSkip();
  OCCA_ASSERT_EQUAL(context.tp.start,
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

void testExpression() {
  tokenContext context;
  exprNode *expr;

  setupContext(context, "");
  OCCA_ASSERT_EQUAL((void*) NULL,
                    (void*) context.getExpression());

  setupContext(context, "1 + 2 + 3");
  expr = context.getExpression();
  OCCA_ASSERT_EQUAL(6,
                    (int) expr->evaluate());
  delete expr;

  expr = context.getExpression(0, 3);
  OCCA_ASSERT_EQUAL(3,
                    (int) expr->evaluate());
  delete expr;

  expr = context.getExpression(2, 5);
  OCCA_ASSERT_EQUAL(5,
                    (int) expr->evaluate());
  delete expr;

  expr = context.getExpression(4, 5);
  OCCA_ASSERT_EQUAL(3,
                    (int) expr->evaluate());
  delete expr;
}
