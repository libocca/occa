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

#include "token.hpp"
#include "tokenizer.hpp"
#include "tokenContext.hpp"

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
  OCCA_ASSERT_EQUAL(0, context.tp.start);
  OCCA_ASSERT_EQUAL(0, context.tp.pos);
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
  OCCA_ASSERT_EQUAL(0, context.tp.pos);
  OCCA_ASSERT_EQUAL(0, context.tp.end);

  context.setup();
  OCCA_ASSERT_EQUAL(0, context.tp.start);
  OCCA_ASSERT_EQUAL(0, context.tp.pos);
  OCCA_ASSERT_EQUAL(3, context.tp.end);

  OCCA_ASSERT_EQUAL((token_t*) newline,
                    context.getNextToken());
  OCCA_ASSERT_EQUAL((token_t*) identifier,
                    context.getNextToken());
  OCCA_ASSERT_EQUAL((token_t*) primitive,
                    context.getNextToken());
  OCCA_ASSERT_EQUAL((token_t*) NULL,
                    context.getNextToken());

  context.push(1, 2);
  OCCA_ASSERT_EQUAL(1, context.tp.start);
  OCCA_ASSERT_EQUAL(1, context.tp.pos);
  OCCA_ASSERT_EQUAL(2, context.tp.end);

  OCCA_ASSERT_EQUAL((token_t*) identifier,
                    context.getNextToken());
  OCCA_ASSERT_EQUAL((token_t*) NULL,
                    context.getNextToken());

  context.pop();
  OCCA_ASSERT_EQUAL(0, context.tp.start);
  OCCA_ASSERT_EQUAL(3, context.tp.pos);
  OCCA_ASSERT_EQUAL(3, context.tp.end);
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
