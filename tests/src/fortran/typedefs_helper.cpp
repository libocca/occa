#include <stdio.h>
#include <occa/c/types.h>

OCCA_START_EXTERN_C

//---[ occaType size ]------------------
static size_t occaTypeGetSize() {
  return sizeof(occaType);
}

static size_t occaTypeMagicHeaderGetSize() {
  return sizeof(((occaType*)0)->magicHeader);
}

static size_t occaTypeTypeGetSize() {
  return sizeof(((occaType*)0)->type);
}

static size_t occaTypeBytesGetSize() {
  return sizeof(((occaType*)0)->bytes);
}

static size_t occaTypeNeedsFreeGetSize() {
  return sizeof(((occaType*)0)->needsFree);
}

static size_t occaTypeValueGetSize() {
  return sizeof(((occaType*)0)->value);
}

bool occaTypeHasCorrectSize(const size_t s,
                            const size_t magicHeader_s,
                            const size_t type_s,
                            const size_t bytes_s,
                            const size_t needsFree_s,
                            const size_t value_s,
                            const bool verbose) {
  const bool ok = (
    s                == occaTypeGetSize()
    && magicHeader_s == occaTypeMagicHeaderGetSize()
    && type_s        == occaTypeTypeGetSize()
    && bytes_s       == occaTypeBytesGetSize()
    && needsFree_s   == occaTypeNeedsFreeGetSize()
    && value_s       == occaTypeValueGetSize()
  );

  if (verbose || (!ok)) {
    printf("\n");
    printf("    ===============================================\n");
    printf("    OCCA C-Fortran interface type size info:\n");
    printf("    -------------------------------+-----+---------\n");
    printf("                                   | C   | Fortran\n");
    printf("    -------------------------------+-----+---------\n");
    printf("      sizeof(occaType)             | %3ld | %3ld\n", occaTypeGetSize(), s);
    printf("      sizeof(occaType.magicHeader) | %3ld | %3ld\n", occaTypeMagicHeaderGetSize(), magicHeader_s);
    printf("      sizeof(occaType.type)        | %3ld | %3ld\n", occaTypeTypeGetSize(), type_s);
    printf("      sizeof(occaType.bytes)       | %3ld | %3ld\n", occaTypeBytesGetSize(), bytes_s);
    printf("      sizeof(occaType.needsFree)   | %3ld | %3ld\n", occaTypeNeedsFreeGetSize(), needsFree_s);
    printf("      sizeof(occaType.value)       | %3ld | %3ld\n", occaTypeValueGetSize(), value_s);
    printf("    ===============================================\n");
    printf("\n");
  }

  return ok;
}
//======================================

//---[ occaType print ]-----------------
void printOccaType(const occaType *value, const bool verbose) {
  printf("\n");
  printf("    ===============================================\n");
  printf("    Dump OCCA type data:\n");
  printf("    -----------------------------------------------\n");
  printf("      Memory address: %p\n", (void*) value);
  printf("\n");
  printf("      magicHeader:    %d\n", value->magicHeader);
  printf("      type:           %d\n", value->type);
  printf("      bytes:          %ld\n", value->bytes);
  printf("      needsFree:      %d (%s)\n", value->needsFree,
         value->needsFree ? "true" : "false");
  printf("      value (C `union`):\n");
  if (verbose) {
    // Print all types of the C union
    printf("      value.uint8_:   %d\n", value->value.uint8_);
    printf("      value.uint16_:  %d\n", value->value.uint16_);
    printf("      value.uint32_:  %d\n", value->value.uint32_);
    printf("      value.uint64_:  %ld\n", value->value.uint64_);
    printf("      value.int8_:    %d\n", value->value.int8_);
    printf("      value.int16_:   %d\n", value->value.int16_);
    printf("      value.int32_:   %d\n", value->value.int32_);
    printf("      value.int64_:   %ld\n", value->value.int64_);
    printf("      value.float_:   %f\n", value->value.float_);
    printf("      value.double_:  %f\n", value->value.double_);
    printf("      value.ptr:      %p\n", (void*) value->value.ptr);
  } else {
    // In the Fortran module we store the data as int64
    printf("      value.int64_:   %ld\n", value->value.int64_);
    printf("      value.ptr:      %p\n", (void*) value->value.ptr);
  }
  printf("    ===============================================\n");
  printf("\n");
}
//======================================

OCCA_END_EXTERN_C
