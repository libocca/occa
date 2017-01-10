#include <sstream>

#include "occa/tools/io.hpp"
#include "occa/tools/json.hpp"
#include "occa/tools/string.hpp"
#include "occa/tools/testing.hpp"

void testString();
void testNumber();
void testObject();
void testArray();
void testKeywords();

int main(const int argc, const char **argv) {
  testString();
  testNumber();
  testObject();
  testArray();
  testKeywords();
}

void testString() {
 occa::json j;

  // Normal strings
  j.load("\"A\"");
  OCCA_TEST_COMPARE(occa::json::string_, j.type);
  OCCA_TEST_COMPARE("A", j.value.string);
  j.load("'A'");
  OCCA_TEST_COMPARE(occa::json::string_, j.type);
  OCCA_TEST_COMPARE("A", j.value.string);
  j.load("\"A'\"");
  OCCA_TEST_COMPARE(occa::json::string_, j.type);
  OCCA_TEST_COMPARE("A'", j.value.string);
  j.load("'A\"'");
  OCCA_TEST_COMPARE(occa::json::string_, j.type);
  OCCA_TEST_COMPARE("A\"", j.value.string);

  // Special chars
  j.load("\"\\\"\"");
  OCCA_TEST_COMPARE(occa::json::string_, j.type);
  OCCA_TEST_COMPARE("\"", j.value.string);
  j.load("\"\\\\\"");
  OCCA_TEST_COMPARE(occa::json::string_, j.type);
  OCCA_TEST_COMPARE("\\", j.value.string);
  j.load("\"\\/\"");
  OCCA_TEST_COMPARE(occa::json::string_, j.type);
  OCCA_TEST_COMPARE("/", j.value.string);
  j.load("\"\\b\"");
  OCCA_TEST_COMPARE(occa::json::string_, j.type);
  OCCA_TEST_COMPARE("\b", j.value.string);
  j.load("\"\\f\"");
  OCCA_TEST_COMPARE(occa::json::string_, j.type);
  OCCA_TEST_COMPARE("\f", j.value.string);
  j.load("\"\\n\"");
  OCCA_TEST_COMPARE(occa::json::string_, j.type);
  OCCA_TEST_COMPARE("\n", j.value.string);
  j.load("\"\\r\"");
  OCCA_TEST_COMPARE(occa::json::string_, j.type);
  OCCA_TEST_COMPARE("\r", j.value.string);
  j.load("\"\\t\"");
  OCCA_TEST_COMPARE(occa::json::string_, j.type);
  OCCA_TEST_COMPARE("\t", j.value.string);
}

void testNumber() {
 occa::json j;

  j.load("-10");
  OCCA_TEST_COMPARE(occa::json::number_, j.type);
  OCCA_TEST_COMPARE(-10, (int) j.value.number);
  j.load("10");
  OCCA_TEST_COMPARE(occa::json::number_, j.type);
  OCCA_TEST_COMPARE(10, (int) j.value.number);
  j.load("0.1");
  OCCA_TEST_COMPARE(occa::json::number_, j.type);
  OCCA_TEST_COMPARE(0.1, (double) j.value.number);
  j.load("0.1e10");
  OCCA_TEST_COMPARE(occa::json::number_, j.type);
  OCCA_TEST_COMPARE(0.1e10, (double) j.value.number);
  j.load("0.1E10");
  OCCA_TEST_COMPARE(occa::json::number_, j.type);
  OCCA_TEST_COMPARE(0.1E10, (double) j.value.number);
  j.load("0.1e-10");
  OCCA_TEST_COMPARE(occa::json::number_, j.type);
  OCCA_TEST_COMPARE(0.1e-10, (double) j.value.number);
  j.load("0.1E-10");
  OCCA_TEST_COMPARE(occa::json::number_, j.type);
  OCCA_TEST_COMPARE(0.1E-10, (double) j.value.number);
  j.load("0.1e+10");
  OCCA_TEST_COMPARE(occa::json::number_, j.type);
  OCCA_TEST_COMPARE(0.1e+10, (double) j.value.number);
  j.load("0.1E+10");
  OCCA_TEST_COMPARE(occa::json::number_, j.type);
  OCCA_TEST_COMPARE(0.1E+10, (double) j.value.number);
}

void testObject() {
 occa::json j;

 j.load("{\"0\":0, \"1\":1}");
 OCCA_TEST_COMPARE(occa::json::object_, j.type);
 OCCA_TEST_COMPARE(2, (int) j.value.object.size());
 OCCA_TEST_COMPARE(0, (int) j.value.object["0"]);
 OCCA_TEST_COMPARE(1, (int) j.value.object["1"]);

 j.load("{\"0\":0, \"1\":1,}");
 OCCA_TEST_COMPARE(occa::json::object_, j.type);
 OCCA_TEST_COMPARE(2, (int) j.value.object.size());
 OCCA_TEST_COMPARE(occa::json::number_, j.value.object["0"].type);
 OCCA_TEST_COMPARE(occa::json::number_, j.value.object["1"].type);
 OCCA_TEST_COMPARE(0, (int) j.value.object["0"]);
 OCCA_TEST_COMPARE(1, (int) j.value.object["1"]);

 // Short-hand notation
 j.load("{0:0, 1:1}");
 OCCA_TEST_COMPARE(occa::json::object_, j.type);
 OCCA_TEST_COMPARE(2, (int) j.value.object.size());
 OCCA_TEST_COMPARE(0, (int) j.value.object["0"]);
 OCCA_TEST_COMPARE(1, (int) j.value.object["1"]);

 j.load("{0:0, 1:1,}");
 OCCA_TEST_COMPARE(occa::json::object_, j.type);
 OCCA_TEST_COMPARE(2, (int) j.value.object.size());
 OCCA_TEST_COMPARE(occa::json::number_, j.value.object["0"].type);
 OCCA_TEST_COMPARE(occa::json::number_, j.value.object["1"].type);
 OCCA_TEST_COMPARE(0, (int) j.value.object["0"]);
 OCCA_TEST_COMPARE(1, (int) j.value.object["1"]);

 // Test path
 j.load("{0: {1: {2: {3: 3}}}}");
 OCCA_TEST_COMPARE(3, (int) j["0/1/2/3"]);
}

void testArray() {
 occa::json j;

 j.load("[1, 2]");
 OCCA_TEST_COMPARE(occa::json::array_, j.type);
 OCCA_TEST_COMPARE(2, (int) j.value.array.size());

 OCCA_TEST_COMPARE(occa::json::number_, j.value.array[0].type);
 OCCA_TEST_COMPARE(occa::json::number_, j.value.array[1].type);
 OCCA_TEST_COMPARE(1, (int) j.value.array[0]);
 OCCA_TEST_COMPARE(2, (int) j.value.array[1]);

 j.load("[1, 2,]");
 OCCA_TEST_COMPARE(occa::json::array_, j.type);
 OCCA_TEST_COMPARE(2, (int) j.value.array.size());

 OCCA_TEST_COMPARE(occa::json::number_, j.value.array[0].type);
 OCCA_TEST_COMPARE(occa::json::number_, j.value.array[1].type);
 OCCA_TEST_COMPARE(1, (int) j.value.array[0]);
 OCCA_TEST_COMPARE(2, (int) j.value.array[1]);
}

void testKeywords() {
 occa::json j;

 j.load("true");
 OCCA_TEST_COMPARE(occa::json::boolean_, j.type);
 OCCA_TEST_COMPARE(true, j.value.boolean);
 j.load("false");
 OCCA_TEST_COMPARE(occa::json::boolean_, j.type);
 OCCA_TEST_COMPARE(false, j.value.boolean);
 j.load("null");
 OCCA_TEST_COMPARE(occa::json::null_, j.type);
}
