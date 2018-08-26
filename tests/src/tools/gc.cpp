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
#include <occa/defines.hpp>
#include <occa/tools/gc.hpp>
#include <occa/tools/testing.hpp>

void testWithRefs();
void testRingEntry();
void testRing();

int main(const int argc, const char **argv) {
  testWithRefs();
  testRingEntry();
  testRing();

  return 0;
}

void testWithRefs() {
  occa::gc::withRefs refs;

  ASSERT_EQ(refs.getRefs(),
            0);

  refs.addRef();
  ASSERT_EQ(refs.getRefs(),
            1);

  refs.removeRef();
  refs.removeRef();
  ASSERT_EQ(refs.getRefs(),
            0);

  refs.dontUseRefs();
  ASSERT_EQ(refs.getRefs(),
            -1);

  refs.addRef();
  ASSERT_EQ(refs.getRefs(),
            -1);

  refs.removeRef();
  ASSERT_EQ(refs.getRefs(),
            -1);

  refs.setRefs(1);
  refs.addRef();
  ASSERT_EQ(refs.getRefs(),
            2);
}

void testRingEntry() {
  occa::gc::ringEntry_t a, b;

  ASSERT_EQ(a.leftRingEntry,
            &a);
  ASSERT_EQ(a.rightRingEntry,
            &a);

  a.removeRef();
  ASSERT_EQ(a.leftRingEntry,
            &a);
  ASSERT_EQ(a.rightRingEntry,
            &a);

  a.leftRingEntry = &b;
  a.rightRingEntry = &b;

  b.leftRingEntry = &a;
  b.rightRingEntry = &a;

  b.removeRef();
  ASSERT_EQ(a.leftRingEntry,
            &a);
  ASSERT_EQ(a.rightRingEntry,
            &a);
}

void testRing() {
  occa::gc::ringEntry_t a, b, c;
  occa::gc::ring_t<occa::gc::ringEntry_t> values;

  ASSERT_EQ((void*) values.head,
            (void*) NULL);
  ASSERT_TRUE(values.needsFree());

  values.addRef(NULL);
  ASSERT_EQ((void*) values.head,
            (void*) NULL);
  ASSERT_TRUE(values.needsFree());

  values.addRef(&a);
  ASSERT_EQ(values.head,
            &a);
  ASSERT_FALSE(values.needsFree());

  values.addRef(&b);
  ASSERT_EQ(values.head,
            &a);
  ASSERT_EQ(values.head->rightRingEntry,
            &b);

  values.removeRef(&a);
  ASSERT_EQ(values.head,
            &b);
  ASSERT_EQ(values.head->rightRingEntry,
            &b);

  values.removeRef(&b);
  ASSERT_EQ((void*) values.head,
            (void*) NULL);
  ASSERT_TRUE(values.needsFree());
}
