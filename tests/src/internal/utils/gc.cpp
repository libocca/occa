#include <occa/defines.hpp>
#include <occa/internal/utils/gc.hpp>
#include <occa/internal/utils/testing.hpp>

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
