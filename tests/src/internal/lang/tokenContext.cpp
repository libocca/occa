#include <sstream>

#include <occa/internal/utils/testing.hpp>

#include <occa/internal/lang/expr.hpp>
#include <occa/internal/lang/token.hpp>
#include <occa/internal/lang/tokenizer.hpp>
#include <occa/internal/lang/tokenContext.hpp>

using namespace occa::lang;

void testMethods();
void testPairs();

std::string source;

void setupContext(tokenContext_t &tokenContext,
                  const std::string &source_) {
  source = source_;
  tokenContext.setup(
    tokenizer_t::tokenize(source)
  );
}

int main(const int argc, const char **argv) {
  testMethods();
  testPairs();

  return 0;
}

void testMethods() {
  tokenContext_t tokenContext;
  ASSERT_EQ(0, tokenContext.tp.start);
  ASSERT_EQ(0, tokenContext.tp.end);

  newlineToken    *newline    = new newlineToken(originSource::string);
  identifierToken *identifier = new identifierToken(originSource::string,
                                                    "identifier");
  primitiveToken  *primitive  = new primitiveToken(originSource::string,
                                                   1, "1");

  tokenVector tokens;
  tokens.push_back(newline);
  tokens.push_back(identifier);
  tokens.push_back(primitive);

  ASSERT_EQ(0, tokenContext.tp.start);
  ASSERT_EQ(0, tokenContext.tp.end);

  tokenContext.setup(tokens);
  ASSERT_EQ(0, tokenContext.tp.start);
  ASSERT_EQ(3, tokenContext.tp.end);

  ASSERT_EQ(3, tokenContext.size());
  ASSERT_EQ((token_t*) newline,
            tokenContext[0]);
  ASSERT_EQ((token_t*) identifier,
            tokenContext[1]);
  ASSERT_EQ((token_t*) primitive,
            tokenContext[2]);

  // Out-of-bounds
  ASSERT_EQ((token_t*) NULL,
            tokenContext[-1]);
  ASSERT_EQ((token_t*) NULL,
            tokenContext[3]);

  tokenContext.push(1, 2);
  ASSERT_EQ(1, tokenContext.tp.start);
  ASSERT_EQ(2, tokenContext.tp.end);

  ASSERT_EQ((token_t*) identifier,
            tokenContext[0]);
  ASSERT_EQ((token_t*) NULL,
            tokenContext[1]);

  tokenRange prev = tokenContext.pop();
  ASSERT_EQ(0, tokenContext.tp.start);
  ASSERT_EQ(3, tokenContext.tp.end);
  ASSERT_EQ(1, prev.start);
  ASSERT_EQ(2, prev.end);

  tokenContext.set(1);
  ASSERT_EQ(1, tokenContext.tp.start);
  ASSERT_EQ(3, tokenContext.tp.end);

  tokenContext.push();
  tokenContext.set(1, 2);
  ASSERT_EQ(2, tokenContext.tp.start);
  ASSERT_EQ(3, tokenContext.tp.end);

  prev = tokenContext.pop();
  ASSERT_EQ(1, tokenContext.tp.start);
  ASSERT_EQ(3, tokenContext.tp.end);
  ASSERT_EQ(1, prev.start);
  ASSERT_EQ(2, prev.end);
}

void testPairs() {
  tokenContext_t tokenContext;
  // 0  | [<<<] [(]
  // 2  |   [[]
  // 3  |     [{] [1] [}] [,] [{] [2] [}]
  // 10 |   []] [,] [[]
  // 13 |     [{] [3] [}] [,] [{] [4] [}]
  // 20 |   []]
  // 21 | [)] [>>>]
  setupContext(tokenContext, "<<<([{1},{2}], [{3},{4}])>>>");
  ASSERT_EQ(8, (int) tokenContext.pairs.size());
  ASSERT_EQ(22, tokenContext.pairs[0]);  // <<<
  ASSERT_EQ(21, tokenContext.pairs[1]);  //  (
  ASSERT_EQ(10, tokenContext.pairs[2]);  //   [
  ASSERT_EQ(5 , tokenContext.pairs[3]);  //    {
  ASSERT_EQ(9 , tokenContext.pairs[7]);  //    {
  ASSERT_EQ(20, tokenContext.pairs[12]); //   [
  ASSERT_EQ(15, tokenContext.pairs[13]); //    {
  ASSERT_EQ(19, tokenContext.pairs[17]); //    {

  // Test pair range pushes
  intIntMap::iterator it = tokenContext.pairs.begin();
  while (it != tokenContext.pairs.end()) {
    const int pairStart = it->first;
    const int pairEnd   = it->second;

    tokenContext.push();
    tokenContext += pairStart;

    tokenContext.pushPairRange();
    ASSERT_EQ(pairEnd - pairStart - 1,
              tokenContext.size());
    tokenContext.pop();
    tokenContext.pop();
    ++it;
  }

  // Test pair range pop
  // [{1}, {2}]
  tokenContext += 2;
  tokenContext.pushPairRange();
  // {1}
  tokenContext.pushPairRange();
  // ,
  tokenContext.popAndSkip();
  ASSERT_EQ_BINARY(tokenType::op,
                   tokenContext[0]->type());
  ASSERT_EQ(operatorType::comma,
            tokenContext[0]->to<operatorToken>().opType());
  // {2}
  ++tokenContext;
  tokenContext.pushPairRange();
  tokenContext.popAndSkip();
  ASSERT_EQ(tokenContext.tp.start,
            tokenContext.tp.end);


  std::cerr << "Testing pair errors:\n";

  setupContext(tokenContext, "1, 2)");
  setupContext(tokenContext, "1, 2]");
  setupContext(tokenContext, "1, 2}");
  setupContext(tokenContext, "1, 2>>>");


  setupContext(tokenContext, "[1, 2)");
  setupContext(tokenContext, "{1, 2]");
  setupContext(tokenContext, "<<<1, 2}");
  setupContext(tokenContext, "(1, 2>>>");
}
