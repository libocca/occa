#include "utils.hpp"

void testBaseTypeLoading();
void testPointerTypeLoading();
void testReferenceTypeLoading();
void testArrayTypeLoading();
void testVariableLoading();
void testArgumentLoading();
void testFunctionPointerLoading();
void testStructLoading();

void testBaseTypeErrors();
void testPointerTypeErrors();
void testArrayTypeErrors();
void testVariableErrors();

int main(const int argc, const char **argv) {
  setupParser();

  testBaseTypeLoading();
  testPointerTypeLoading();
  testReferenceTypeLoading();
  testArrayTypeLoading();
  testVariableLoading();
  testArgumentLoading();
  testFunctionPointerLoading();
  testStructLoading();

  std::cerr << "\n---[ Testing type errors ]----------------------\n\n";
  testBaseTypeErrors();
  testPointerTypeErrors();
  testArrayTypeErrors();
  testVariableErrors();
  std::cerr << "================================================\n\n";

  return 0;
}

vartype_t loadType(const std::string &s) {
  setSource(s);
  return parser.loadType();
}

#define assertType(str_)                            \
  setSource(str_);                                  \
  parser.loadType();                                \
  ASSERT_FALSE(parser.isLoadingFunctionPointer());  \
  ASSERT_FALSE(parser.isLoadingVariable())

vartype_t loadVariableType(const std::string &s) {
  setSource(s);
  return parser.loadVariable().vartype;
}

#define assertVariable(str_)                        \
  setSource(str_);                                  \
  parser.loadType();                                \
  ASSERT_FALSE(parser.isLoadingFunctionPointer());  \
  ASSERT_TRUE(parser.isLoadingVariable())

variable_t loadVariable(const std::string &s) {
  setSource(s);
  return parser.loadVariable();
}

#define assertFunctionPointer(str_)               \
  setSource(str_);                                \
  parser.loadType();                              \
  ASSERT_TRUE(parser.isLoadingFunctionPointer())

void testBaseTypeLoading() {
  vartype_t type;

  // Test base type
  type = loadType("int");
  ASSERT_EQ(0,
            type.qualifiers.size());
  ASSERT_EQ(&int_,
            type.type);

  type = loadType("const volatile float");
  ASSERT_EQ(2,
            type.qualifiers.size());
  ASSERT_TRUE(type.has(volatile_));
  ASSERT_TRUE(type.has(const_));
  ASSERT_EQ(&float_,
            type.type);

  type = loadType("const long long");
  ASSERT_EQ(2,
            type.qualifiers.size());
  ASSERT_TRUE(type.has(const_));
  ASSERT_TRUE(type.has(longlong_));
  ASSERT_EQ(&int_,
            type.type);

  // Test weird order declaration
  type = loadType("double const long long");
  ASSERT_EQ(2,
            type.qualifiers.size());
  ASSERT_TRUE(type.has(const_));
  ASSERT_TRUE(type.has(longlong_));
  ASSERT_EQ(&double_,
            type.type);
}

void testPointerTypeLoading() {
  vartype_t type;

  type = loadType("int *");
  ASSERT_EQ(1,
            (int) type.pointers.size());
  ASSERT_EQ(0,
            type.pointers[0].qualifiers.size());

  type = loadType("const volatile float * const");
  ASSERT_EQ(1,
            (int) type.pointers.size());
  ASSERT_EQ(1,
            type.pointers[0].qualifiers.size());
  ASSERT_TRUE(type.pointers[0].has(const_));

  type = loadType("float * const * volatile ** const volatile");
  ASSERT_EQ(4,
            (int) type.pointers.size());
  ASSERT_TRUE(type.pointers[0].has(const_));
  ASSERT_TRUE(type.pointers[1].has(volatile_));
  ASSERT_EQ(0,
            type.pointers[2].qualifiers.size());
  ASSERT_TRUE(type.pointers[3].has(const_));
  ASSERT_TRUE(type.pointers[3].has(volatile_));
}

void testReferenceTypeLoading() {
  vartype_t type;

  type = loadType("int");
  ASSERT_FALSE(type.isReference());
  type = loadType("int &");
  ASSERT_TRUE(type.isReference());

  type = loadType("int *");
  ASSERT_FALSE(type.isReference());
  type = loadType("int *&");
  ASSERT_TRUE(type.isReference());

  type = loadType("int ***");
  ASSERT_FALSE(type.isReference());
  type = loadType("int ***&");
  ASSERT_TRUE(type.isReference());
}

void testArrayTypeLoading() {
  vartype_t type;

  assertType("int[]");
  type = loadVariableType("int[]");
  ASSERT_EQ(1,
            (int) type.arrays.size());

  assertType("int[][]");
  type = loadVariableType("int[][]");
  ASSERT_EQ(2,
            (int) type.arrays.size());

  assertType("int[1]");
  type = loadVariableType("int[1]");
  ASSERT_EQ(1,
            (int) type.arrays.size());
  ASSERT_EQ(1,
            (int) type.arrays[0].evaluateSize());

  assertType("int[1 + 3][7]");
  type = loadVariableType("int[1 + 3][7]");
  ASSERT_EQ(2,
            (int) type.arrays.size());
  ASSERT_EQ(4,
            (int) type.arrays[0].evaluateSize());
  ASSERT_EQ(7,
            (int) type.arrays[1].evaluateSize());
}

void testVariableLoading() {
  variable_t var;
  std::string varName;

  assertVariable("int varname[]");
  var = loadVariable("int varname[]");
  varName = var.name();
  ASSERT_EQ("varname",
            varName);
  ASSERT_EQ(1,
            (int) var.vartype.arrays.size());

  assertVariable("int varname[][]");
  var = loadVariable("int varname[][]");
  varName = var.name();
  ASSERT_EQ("varname",
            varName);
  ASSERT_EQ(2,
            (int) var.vartype.arrays.size());

  assertVariable("int varname[1]");
  var = loadVariable("int varname[1]");
  varName = var.name();
  ASSERT_EQ("varname",
            varName);
  ASSERT_EQ(1,
            (int) var.vartype.arrays.size());
  ASSERT_EQ(1,
            (int) var.vartype.arrays[0].evaluateSize());

  assertVariable("int varname[1 + 3][7]");
  var = loadVariable("int varname[1 + 3][7]");
  varName = var.name();
  ASSERT_EQ("varname",
            varName);
  ASSERT_EQ(2,
            (int) var.vartype.arrays.size());
  ASSERT_EQ(4,
            (int) var.vartype.arrays[0].evaluateSize());
  ASSERT_EQ(7,
            (int) var.vartype.arrays[1].evaluateSize());
}

void testArgumentLoading() {
  // Test argument detection
  tokenRangeVector argRanges;

  setSource("");
  getArgumentRanges(parser.tokenContext,
                    argRanges);
  ASSERT_EQ(0,
            (int) argRanges.size());

  setSource("a, b");
  getArgumentRanges(parser.tokenContext,
                    argRanges);
  ASSERT_EQ(2,
            (int) argRanges.size());

  setSource("(,,)");
  getArgumentRanges(parser.tokenContext,
                    argRanges);
  ASSERT_EQ(1,
            (int) argRanges.size());

  setSource("(,,), (,,), (,,)");
  getArgumentRanges(parser.tokenContext,
                    argRanges);
  ASSERT_EQ(3,
            (int) argRanges.size());

  // Removes trailing comma
  setSource("a, b,");
  getArgumentRanges(parser.tokenContext,
                    argRanges);
  ASSERT_EQ(2,
            (int) argRanges.size());

  // Test arguments
}

void testFunctionPointerLoading() {
  variable_t var;
  std::string varName;

#define varFunc var.vartype.type->to<functionPtr_t>()

  // Test pointer vs block
  assertFunctionPointer("int (*varname)()");
  var = loadVariable("int (*varname)()");

  ASSERT_EQ_BINARY(typeType::functionPtr,
                   var.vartype.type->type());
  varName = var.name();
  ASSERT_EQ("varname",
            varName);
  ASSERT_FALSE(varFunc.isBlock);

  assertFunctionPointer("int (^varname)()");
  var = loadVariable("int (^varname)()");
  varName = var.name();
  ASSERT_EQ("varname",
            varName);
  ASSERT_TRUE(varFunc.isBlock);

  // Test arguments
  var = loadVariable("int (*varname)()");
  ASSERT_EQ(0,
            (int) varFunc.args.size());

  var = loadVariable("int (*varname)(const int i = 0,)");
  ASSERT_EQ(1,
            (int) varFunc.args.size());
  ASSERT_EQ(&int_,
            varFunc.args[0].vartype.type);
  ASSERT_TRUE(varFunc.args[0].vartype.has(const_));
  ASSERT_EQ("i",
            varFunc.args[0].name());

  var = loadVariable("int (*varname)(int, double,)");
  ASSERT_EQ(2,
            (int) varFunc.args.size());
  ASSERT_EQ(&int_,
            varFunc.args[0].vartype.type);
  ASSERT_EQ(&double_,
            varFunc.args[1].vartype.type);

#undef varFunc
}

void testStructLoading() {
  vartype_t type;

  type = loadType("struct foo1 {}");
  ASSERT_EQ("foo1", type.name());
  ASSERT_TRUE(type.has(struct_));

  type = loadType("struct foo2 {} bar2");
  ASSERT_EQ("foo2", type.name());
  ASSERT_TRUE(type.has(struct_));

  type = loadType("struct {} bar3");
  ASSERT_EQ(0, (int) type.name().size());
  ASSERT_TRUE(type.has(struct_));

  type = loadType("typedef struct foo4 {} bar4");
  ASSERT_EQ("bar4", type.name());
  ASSERT_TRUE(type.has(typedef_));

  vartype_t foo4 = ((typedef_t*) type.type)->baseType;
  ASSERT_EQ("foo4", foo4.name());
  ASSERT_TRUE(foo4.has(struct_));
}

void testBaseTypeErrors() {
  vartype_t type;
  type = loadType("const");
  type = loadType("const foo");
  type = loadType("const const");
  type = loadType("long long long");
}

void testPointerTypeErrors() {
  vartype_t type;
  type = loadType("const *");
  type = loadType("float * long");
}

void testArrayTypeErrors() {
  assertType("int[-]");
  loadVariableType("int[-]");
}

void testVariableErrors() {
  assertVariable("int varname[-]");
  loadVariable("int varname[-]");
}
