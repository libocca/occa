#include "utils.hpp"

void testStringMethods();
void testStringMerging();

using namespace occa::lang;

int main(const int argc, const char **argv) {
  testStringMethods();
  testStringMerging();

  return 0;
}

void addEncodingPrefixes(std::string &s1,
                         std::string &s2,
                         std::string &s3,
                         const std::string &encoding1,
                         const std::string &encoding2) {
  const std::string a  = "\"a\"";
  const std::string b  = "\"b\"";
  s1 = (encoding1 + a) + " " + b;
  s2 = a               + " " + (encoding2 + b);
  s3 = (encoding1 + a) + " " + (encoding2 + b);
}

void testStringMethods() {
  // Test values
  testStringValue("\"\""                , "");
  testStringValue("\"\\\"\""            , "\"");
  testStringValue("\"string\\\"string\"", "string\"string");
  testStringValue("R\"(string)\""       , "string");
  testStringValue("u8\"string\""        , "string");
  testStringValue("u\"string\""         , "string");
  testStringValue("U\"string\""         , "string");
  testStringValue("L\"string\""         , "string");
  testStringValue("u8R\"(string)\""     , "string");
  testStringValue("uR\"(string)\""      , "string");
  testStringValue("UR\"(string)\""      , "string");
  testStringValue("LR\"(string)\""      , "string");

  // Test raw strings
  testStringValue("R\"*(string)*\""  , "string");
  testStringValue("u8R\"*(string)*\"", "string");
  testStringValue("uR\"*(string)*\"" , "string");
  testStringValue("UR\"*(string)*\"" , "string");
  testStringValue("LR\"*(string)*\"" , "string");

  testStringValue("R\"foo(string)foo\""  , "string");
  testStringValue("u8R\"foo(string)foo\"", "string");
  testStringValue("uR\"foo(string)foo\"" , "string");
  testStringValue("UR\"foo(string)foo\"" , "string");
  testStringValue("LR\"foo(string)foo\"" , "string");
}

void testStringMerging() {
  // Test string concatination
  const std::string ab = "ab";
  std::string s1, s2, s3;
  addEncodingPrefixes(s1, s2, s3, "", "");
  testStringMergeValue(s1, ab);
  testStringMergeValue(s2, ab);
  testStringMergeValue(s3, ab);
  testStringMergeToken(s1, 0);
  testStringMergeToken(s2, 0);
  testStringMergeToken(s3, 0);

  addEncodingPrefixes(s1, s2, s3, "", "u");
  testStringMergeValue(s1, ab);
  testStringMergeValue(s2, ab);
  testStringMergeValue(s3, ab);
  testStringMergeToken(s1, 0);
  testStringMergeToken(s2, encodingType::u);
  testStringMergeToken(s3, encodingType::u);

  addEncodingPrefixes(s1, s2, s3, "", "U");
  testStringMergeValue(s1, ab);
  testStringMergeValue(s2, ab);
  testStringMergeValue(s3, ab);
  testStringMergeToken(s1, 0);
  testStringMergeToken(s2, encodingType::U);
  testStringMergeToken(s3, encodingType::U);

  addEncodingPrefixes(s1, s2, s3, "", "L");
  testStringMergeValue(s1, ab);
  testStringMergeValue(s2, ab);
  testStringMergeValue(s3, ab);
  testStringMergeToken(s1, 0);
  testStringMergeToken(s2, encodingType::L);
  testStringMergeToken(s3, encodingType::L);

  addEncodingPrefixes(s1, s2, s3, "u", "U");
  testStringMergeValue(s1, ab);
  testStringMergeValue(s2, ab);
  testStringMergeValue(s3, ab);
  testStringMergeToken(s1, encodingType::u);
  testStringMergeToken(s2, encodingType::U);
  testStringMergeToken(s3, encodingType::U);

  addEncodingPrefixes(s1, s2, s3, "u", "L");
  testStringMergeValue(s1, ab);
  testStringMergeValue(s2, ab);
  testStringMergeValue(s3, ab);
  testStringMergeToken(s1, encodingType::u);
  testStringMergeToken(s2, encodingType::L);
  testStringMergeToken(s3, encodingType::L);

  addEncodingPrefixes(s1, s2, s3, "U", "L");
  testStringMergeValue(s1, ab);
  testStringMergeValue(s2, ab);
  testStringMergeValue(s3, ab);
  testStringMergeToken(s1, encodingType::U);
  testStringMergeToken(s2, encodingType::L);
  testStringMergeToken(s3, encodingType::L);

  testStringMergeValue("\"a\" \n \"b\"\n\"c\"\"d\"",
                       "abcd");
}
