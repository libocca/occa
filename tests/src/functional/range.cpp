#include <occa.hpp>
#include <occa/internal/utils/testing.hpp>

void testBaseMethods();
void testEvery();
void testSome();
void testFindIndex();
void testForEach();
void testMap();
void testMapTo();
void testReduce();
void testToArray();

int main(const int argc, const char **argv) {
  occa::setDevice(occa::host());

  testBaseMethods();
  testEvery();
  testSome();
  testFindIndex();
  testForEach();
  testMap();
  testMapTo();
  testReduce();
  testToArray();

  return 0;
}

void testBaseMethods() {
  {
    occa::range range(10);
    ASSERT_EQ(0, range.start);
    ASSERT_EQ(10, range.end);
    ASSERT_EQ(1, range.step);
    ASSERT_EQ(10, (int) range.length());
  }

  {
    occa::range range(-10);
    ASSERT_EQ(0, range.start);
    ASSERT_EQ(-10, range.end);
    ASSERT_EQ(-1, range.step);
    ASSERT_EQ(10, (int) range.length());
  }

  {
    occa::range range(2, 10);
    ASSERT_EQ(2, range.start);
    ASSERT_EQ(10, range.end);
    ASSERT_EQ(1, range.step);
    ASSERT_EQ(8, (int) range.length());
  }

  {
    occa::range range(10, 2);
    ASSERT_EQ(10, range.start);
    ASSERT_EQ(2, range.end);
    ASSERT_EQ(-1, range.step);
    ASSERT_EQ(8, (int) range.length());
  }

  {
    occa::range range(2, 10, 7);
    ASSERT_EQ(2, range.start);
    ASSERT_EQ(10, range.end);
    ASSERT_EQ(7, range.step);
    ASSERT_EQ(2, (int) range.length());
  }

  {
    occa::range range(2, 10, 8);
    ASSERT_EQ(2, range.start);
    ASSERT_EQ(10, range.end);
    ASSERT_EQ(8, range.step);
    ASSERT_EQ(1, (int) range.length());
  }

  {
    occa::range range(2, 10, -1);
    ASSERT_EQ(2, range.start);
    ASSERT_EQ(10, range.end);
    ASSERT_EQ(-1, range.step);
    ASSERT_EQ(0, (int) range.length());
  }

  {
    occa::range range(2, 10, 0);
    ASSERT_EQ(2, range.start);
    ASSERT_EQ(10, range.end);
    ASSERT_EQ(1, range.step);
    ASSERT_EQ(8, (int) range.length());
  }
}

void testEvery() {
  ASSERT_TRUE(
    occa::range(10)
      .every(OCCA_FUNCTION([](const int index) -> bool {
        return index >= 0;
      }))
  );

  ASSERT_TRUE(
    occa::range(-9, -1)
      .every(OCCA_FUNCTION([](const int index) -> bool {
        return index <= 0;
      }))
  );

  ASSERT_FALSE(
    occa::range(10)
      .every(OCCA_FUNCTION([](const int index) -> bool {
        return index >= 1;
      }))
  );
}

void testSome() {
  ASSERT_TRUE(
    occa::range(10)
      .some(OCCA_FUNCTION([](const int index) -> bool {
        return index <= 0;
      }))
  );

  ASSERT_TRUE(
    occa::range(-9, 0)
      .some(OCCA_FUNCTION([](const int index) -> bool {
        return index >= -1;
      }))
  );

  ASSERT_FALSE(
    occa::range(10)
      .some(OCCA_FUNCTION([](const int index) -> bool {
        return index < 0;
      }))
  );
}

void testFindIndex() {
  ASSERT_EQ(
    5, (
      occa::range(10)
      .findIndex(OCCA_FUNCTION([](const int index) -> bool {
        return index == 5;
      }))
    )
  );

  ASSERT_EQ(
    1, (
      occa::range(0, 10, 4)
      .findIndex(OCCA_FUNCTION([](const int index) -> bool {
        return index == 4;
      }))
    )
  );

  ASSERT_EQ(
    9, (
      occa::range(-10, 0)
      .findIndex(OCCA_FUNCTION([](const int index) -> bool {
        return index == -1;
      }))
    )
  );

  ASSERT_EQ(
    -1, (
      occa::range(10)
      .findIndex(OCCA_FUNCTION([](const int index) -> bool {
        return index == -1;
      }))
    )
  );
}

void testForEach() {
  occa::range(10)
    .forEach(OCCA_FUNCTION([](const int index) -> void {
      // Do nothing
    }));
}

void testMap() {
  occa::array<float> floatArray = (
    occa::range(10)
    .map<float>(OCCA_FUNCTION([](const int index) -> float {
      return index / 2.0;
    }))
  );

  ASSERT_EQ(10, (int) floatArray.length());
  ASSERT_EQ((float) 0.0, floatArray.min());
  ASSERT_EQ((float) 4.5, floatArray.max());
}

void testMapTo() {
  occa::array<float> floatArray(10);

  occa::range(10)
    .mapTo<float>(floatArray, OCCA_FUNCTION([](const int index) -> float {
      return index / 2.0;
    }));

  ASSERT_EQ(10, (int) floatArray.length());
  ASSERT_EQ((float) 0.0, floatArray.min());
  ASSERT_EQ((float) 4.5, floatArray.max());
}

void testReduce() {
  const int length = 10;

  int addReduction = 0;
  float subtractReduction = 0;
  long multiplyReduction = 0;
  int boolOrReduction = 0;
  int boolAndReduction = 0;
  int minReduction = 0;
  int maxReduction = 0;

  for (int index = 0; index < length; ++index) {
    addReduction += index;
    subtractReduction -= index;
    multiplyReduction *= index;
    boolOrReduction = boolOrReduction || index;
    boolAndReduction = boolAndReduction && index;
    minReduction = std::min(minReduction, index);
    maxReduction = std::max(maxReduction, index);
  }

  ASSERT_EQ(
    addReduction,
    occa::range(length)
    .reduce<int>(
      occa::reductionType::sum,
      OCCA_FUNCTION([](const int &acc, const int index) -> int {
        return acc + index;
      })
    )
  );

  ASSERT_EQ(
    subtractReduction,
    occa::range(length)
    .reduce<float>(
      occa::reductionType::sum,
      OCCA_FUNCTION([](const float &acc, const int index) -> float {
        return acc - index;
      })
    )
  );

  ASSERT_EQ(
    multiplyReduction,
    occa::range(length)
    .reduce<int>(
      occa::reductionType::multiply,
      OCCA_FUNCTION([](const int &acc, const int index) -> int {
        return acc * index;
      })
    )
  );

  ASSERT_EQ(
    boolOrReduction,
    occa::range(length)
    .reduce<bool>(
      occa::reductionType::boolOr,
      OCCA_FUNCTION([](const bool &acc, const int index) -> bool {
        return acc || index;
      })
    )
  );

  ASSERT_EQ(
    boolAndReduction,
    occa::range(length)
    .reduce<bool>(
      occa::reductionType::boolAnd,
      OCCA_FUNCTION([](const bool &acc, const int index) -> bool {
        return acc && index;
      })
    )
  );

  ASSERT_EQ(
    minReduction,
    occa::range(length)
    .reduce<int>(
      occa::reductionType::min,
      OCCA_FUNCTION([](const int &acc, const int index) -> int {
        return acc < index ? acc : index;
      })
    )
  );

  ASSERT_EQ(
    maxReduction,
    occa::range(length)
    .reduce<int>(
      occa::reductionType::max,
      OCCA_FUNCTION([](const int &acc, const int index) -> int {
        return acc > index ? acc : index;
      })
    )
  );
}

void testToArray() {
  occa::array<int> arr = occa::range(10).toArray();
  ASSERT_EQ(10, (int) arr.length());
  ASSERT_EQ(0, arr.min());
  ASSERT_EQ(9, arr.max());

  arr = occa::range(0, -10).toArray();
  ASSERT_EQ(10, (int) arr.length());
  ASSERT_EQ(-9, arr.min());
  ASSERT_EQ(0, arr.max());

  arr = occa::range(0, 10, 2).toArray();
  ASSERT_EQ(5, (int) arr.length());
  ASSERT_EQ(0, arr.min());
  ASSERT_EQ(8, arr.max());
}
