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

#define OCCA_DISABLE_VARIADIC_MACROS

#include <occa.hpp>
#include <occa.h>
#include <occa/c/types.hpp>
#include <occa/tools/testing.hpp>

void testInit();
void testUvaMethods();
void testCopyMethods();
void testInteropMethods();

int main(const int argc, const char **argv) {
  testInit();
  testUvaMethods();
  testCopyMethods();
  testInteropMethods();

  return 0;
}

void testInit() {
  const size_t bytes = 3 * sizeof(int);
  int *data = new int[3];
  data[0] = 0;
  data[1] = 1;
  data[2] = 2;

  occaMemory mem = occaUndefined;
  occaProperties props = (
    occaCreatePropertiesFromString("foo: 'bar'")
  );

  ASSERT_TRUE(occaIsUndefined(mem));
  ASSERT_EQ(mem.type,
            OCCA_UNDEFINED);
  ASSERT_FALSE(occaMemoryIsInitialized(mem));

  mem = occaMalloc(bytes, data, props);
  ASSERT_FALSE(occaIsUndefined(mem));
  ASSERT_EQ(mem.type,
            OCCA_MEMORY);
  ASSERT_TRUE(occaMemoryIsInitialized(mem));

  int *ptr = (int*) occaMemoryPtr(mem, occaDefault);
  ASSERT_EQ(ptr[0], 0);
  ASSERT_EQ(ptr[1], 1);
  ASSERT_EQ(ptr[2], 2);

  int *ptr2 = (int*) occaMemoryPtr(mem, props);
  ASSERT_EQ(ptr, ptr2);

  ASSERT_EQ(occa::c::device(occaMemoryGetDevice(mem)),
            occa::host());

  occaProperties memProps = occaMemoryGetProperties(mem);
  occaType memMode = occaPropertiesGet(memProps, "foo", occaUndefined);
  ASSERT_EQ((const char*) occaJsonGetString(memMode),
            (const char*) "bar");

  ASSERT_EQ((size_t) occaMemorySize(mem),
            bytes);

  occaMemory subMem = occaMemorySlice(mem,
                                      1 * sizeof(int),
                                      occaAllBytes);

  ASSERT_EQ((size_t) occaMemorySize(subMem),
            bytes - (1 * sizeof(int)));

  ptr = (int*) occaMemoryPtr(subMem, occaDefault);
  ASSERT_EQ(ptr[0], 1);
  ASSERT_EQ(ptr[1], 2);

  occaFree(props);
  occaFree(subMem);
  occaFree(mem);

  delete [] data;
}

void testUvaMethods() {
  // Test with uninitialized memory
  occaMemory mem = occaUndefined;

  ASSERT_FALSE(occaMemoryIsManaged(mem));
  ASSERT_FALSE(occaMemoryInDevice(mem));
  ASSERT_FALSE(occaMemoryIsStale(mem));

  occaMemoryStartManaging(mem);
  ASSERT_FALSE(occaMemoryIsManaged(mem));

  occaMemoryStopManaging(mem);
  ASSERT_FALSE(occaMemoryIsManaged(mem));

  ASSERT_THROW(
    occaMemorySyncToDevice(mem, occaAllBytes, 0);
  );

  ASSERT_THROW(
    occaMemorySyncToHost(mem, occaAllBytes, 0);
  );

  // Test with memory
  mem = occaMalloc(10 * sizeof(int), NULL, occaDefault);

  ASSERT_FALSE(occaMemoryIsManaged(mem));
  ASSERT_FALSE(occaMemoryInDevice(mem));
  ASSERT_FALSE(occaMemoryIsStale(mem));

  occaMemoryStartManaging(mem);
  ASSERT_TRUE(occaMemoryIsManaged(mem));

  occaMemoryStopManaging(mem);
  ASSERT_FALSE(occaMemoryIsManaged(mem));

  occaMemorySyncToDevice(mem, occaAllBytes, 0);
  occaMemorySyncToHost(mem, occaAllBytes, 0);

  occaFree(mem);
}

void testCopyMethods() {
  const size_t bytes2 = 2 * sizeof(int);
  int *data2 = new int[2];
  data2[0] = 0;
  data2[1] = 1;

  const size_t bytes4 = 4 * sizeof(int);
  int *data4 = new int[4];
  data4[0] = 0;
  data4[1] = 1;
  data4[2] = 2;
  data4[3] = 3;

  occaMemory mem2 = occaMalloc(bytes2, data2, occaDefault);
  occaMemory mem4 = occaMalloc(bytes4, data4, occaDefault);

  occaProperties props = (
    occaCreatePropertiesFromString("foo: 'bar'")
  );

  int *ptr2 = (int*) occaMemoryPtr(mem2, occaDefault);
  int *ptr4 = (int*) occaMemoryPtr(mem4, occaDefault);

  // Mem -> Mem
  // Copy over [2, 3]
  occaCopyMemToMem(mem2, mem4,
                   bytes2,
                   0, bytes2,
                   occaDefault);

  ASSERT_EQ(ptr2[0], 2);
  ASSERT_EQ(ptr2[1], 3);

  // Copy over [2] to the end
  occaCopyMemToMem(mem4, mem2,
                   1 * sizeof(int),
                   3 * sizeof(int), 0,
                   props);

  ASSERT_EQ(ptr4[0], 0);
  ASSERT_EQ(ptr4[1], 1);
  ASSERT_EQ(ptr4[2], 2);
  ASSERT_EQ(ptr4[3], 2);

  // Ptr <-> Mem with default props
  occaCopyPtrToMem(mem4, data4,
                   occaAllBytes, 0,
                   occaDefault);
  ASSERT_EQ(ptr4[3], 3);
  ptr4[3] = 2;

  occaCopyMemToPtr(data4, mem4,
                   occaAllBytes, 0,
                   occaDefault);
  ASSERT_EQ(data4[3], 2);

  // Ptr <-> Mem with props
  occaCopyMemToPtr(data2, mem2,
                   occaAllBytes, 0,
                   props);

  ASSERT_EQ(data2[0], 2);
  ASSERT_EQ(data2[1], 3);
  data2[1] = 1;

  occaCopyPtrToMem(mem2, data2,
                   occaAllBytes, 0,
                   props);
  ASSERT_EQ(ptr2[1], 1);

  occaFree(mem2);
  occaFree(mem4);

  // UVA memory copy
  int *o_data2 = (int*) occaUMalloc(bytes2, data2, occaDefault);
  int *o_data4 = (int*) occaUMalloc(bytes4, data4, occaDefault);

  o_data2[0] = -2;
  o_data4[1] = -4;

  occaMemcpy(o_data2, o_data4 + 1,
             1 * sizeof(int),
             occaDefault);

  occaMemcpy(data2, o_data2,
             occaAllBytes,
             props);

  ASSERT_EQ(data2[0], -4);

  o_data2[0] = 1;
  occaMemcpy(o_data2, data2,
             occaAllBytes,
             props);

  ASSERT_EQ(o_data2[0], -4);

  // Unable to find 'all bytes' from 2 non-occa pointers
  ASSERT_THROW(
    occaMemcpy(data2, data4,
               occaAllBytes,
               occaDefault);
  );

  delete [] data2;
  delete [] data4;
  occaFreeUvaPtr(o_data2);
  occaFreeUvaPtr(o_data4);
  occaFree(props);
}

void testInteropMethods() {
  const int entries = 10;
  const size_t bytes = entries * sizeof(int);

  occaMemory mem1 = occaMalloc(bytes, NULL, occaDefault);
  occaMemory mem2 = occaMemoryClone(mem1);

  ASSERT_EQ(occaMemorySize(mem1),
            occaMemorySize(mem2));

  ASSERT_NEQ(occa::c::memory(mem1),
             occa::c::memory(mem2));

  int *ptr = (int*) occaMemoryPtr(mem2, occaDefault);
  occaMemoryDetach(mem2);

  for (int i = 0; i < entries; ++i) {
    ptr[i] = i;
  }

  mem2 = occaWrapCpuMemory(occaHost(),
                           ptr,
                           bytes,
                           occaDefault);

  occaProperties props = (
    occaCreatePropertiesFromString("foo: 'bar'")
  );
  mem2 = occaWrapCpuMemory(occaHost(),
                           ptr,
                           bytes,
                           props);
  occaFree(mem1);
  occaFree(mem2);
  occaFree(props);
}
