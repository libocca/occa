#include <occa.hpp>
#include <occa/functional.hpp>
#include <occa/internal/functional/functionStore.hpp>
#include <occa/internal/utils/testing.hpp>

class context {
 public:
  occa::device device;

  int length;
  int minValue;
  int maxValue;

  int *values;
  occa::memory memory;
  occa::array<int> array;

  context(occa::device &device_) :
    device(device_) {
    length = 10;
    minValue = 0;
    maxValue = 9;

    values = new int[10];
    for (int i = 0; i < length; ++i) {
      values[i] = i;
    }

    memory = device.malloc<int>(length, values);
    array = occa::array<int>(memory);
  }

  ~context() {
    delete [] values;
  }
};

void testFunctionStore();
void testBaseMethods(occa::device device);
void testEvery(occa::device device);
void testSome(occa::device device);
void testFilter(occa::device device);
void testFindIndex(occa::device device);
void testForEach(occa::device device);
void testMap(occa::device device);
void testMapTo(occa::device device);
void testReduce(occa::device device);
void testSlice(occa::device device);
void testConcat(occa::device device);
void testFill(occa::device device);
void testIncludes(occa::device device);
void testIndexOf(occa::device device);
void testLastIndexOf(occa::device device);
void testCast(occa::device device);
void testReverse(occa::device device);
void testShiftLeft(occa::device device);
void testShiftRight(occa::device device);
void testMax(occa::device device);
void testMin(occa::device device);
void testDotProduct(occa::device device);
void testClamp(occa::device device);

int main(const int argc, const char **argv) {
  std::vector<occa::device> devices = {
    occa::device({
      {"mode", "Serial"}
    }),
    occa::device({
      {"mode", "OpenMP"}
    }),
    occa::device({
      {"mode", "CUDA"},
      {"device_id", 0}
    }),
    occa::device({
      {"mode", "HIP"},
      {"device_id", 0}
    })
  };

  testFunctionStore();

  for (auto &device : devices) {
    std::cout << "Testing mode: " << device.mode() << '\n';
    testBaseMethods(device);
    testEvery(device);
    testSome(device);
    testFilter(device);
    testFindIndex(device);
    testForEach(device);
    testMap(device);
    testMapTo(device);
    testReduce(device);
    testSlice(device);
    testConcat(device);
    testFill(device);
    testIncludes(device);
    testIndexOf(device);
    testLastIndexOf(device);
    testCast(device);
    testReverse(device);
    testShiftLeft(device);
    testShiftRight(device);
    testMax(device);
    testMin(device);
    testDotProduct(device);
    testClamp(device);
  }

  return 0;
}

void testFunctionStore() {
  // Test that we inserted a new function in the functionStore only once
  const size_t initialStoreSize = occa::functionStore.size();
  for (int i = 0; i < 10; ++i) {
    occa::function<bool(int, int, const int*)> func = (
      OCCA_FUNCTION([](const int entry, const int index, const int *entries) -> bool {
        return false;
      })
    );
  }
  ASSERT_EQ(initialStoreSize + 1,
            occa::functionStore.size());

  // Test arguments and defines
  const int value = 1;
  const int multiplier = 2;

  occa::scope scope({
    // Passed as arguments
    {"value", value}
  }, {
    // Passed as compile-time #defines
    {"defines/multiplier", multiplier}
  });

  auto func1 = OCCA_FUNCTION(scope, [](const int entry) -> int {
    return value + (entry * multiplier);
  });
  auto func2 = OCCA_FUNCTION(scope, [](const int entry) -> int {
    return value + (entry * multiplier);
  });
  // These two functions should be identical so we should only be incrementing
  // the store size by 1
  ASSERT_EQ(initialStoreSize + 2,
            occa::functionStore.size());

  // WARNING: This is invalid because OKL won't be able to generate the
  //          valid source without entry and multiplier
  auto func3 = OCCA_FUNCTION([](const int entry) -> int {
    return value + (entry * multiplier);
  });
  // The scope changed so it should generate a new hash
  ASSERT_EQ(initialStoreSize + 3,
            occa::functionStore.size());

  // Test the lambda call
  ASSERT_EQ(7, func1(3));
}

void testBaseMethods(occa::device device) {
  context ctx(device);

  ASSERT_EQ(ctx.length,
            (int) ctx.array.length());

  ASSERT_EQ(ctx.device,
            ctx.array.getDevice());

  ASSERT_EQ(ctx.array.memory(),
            occa::array<int>(ctx.array).memory());

  ASSERT_EQ(0, ctx.array[0]);
  ASSERT_EQ(4, ctx.array[4]);

  ctx.array.resize(5);
  ASSERT_EQ(5, (int) ctx.array.length());
  ASSERT_EQ(0, ctx.array[0]);
  ASSERT_EQ(4, ctx.array[4]);

  ctx.array.resize(10);
  ASSERT_EQ(10, (int) ctx.array.length());
  ASSERT_EQ(0, ctx.array[0]);
  ASSERT_EQ(4, ctx.array[4]);
}

void testEvery(occa::device device) {
  context ctx(device);

  ASSERT_TRUE(
    ctx.array
      .every(OCCA_FUNCTION([](const int &value) -> bool {
        return value >= 0;
      }))
  );

  ASSERT_TRUE(
    ctx.array
      .every(OCCA_FUNCTION([](const int &value, const int index) -> bool {
        return value >= 0;
      }))
  );

  ASSERT_TRUE(
    ctx.array
    .every(OCCA_FUNCTION([](const int &value, const int index, const int *values) -> bool {
      return (
        (value == values[index]) &&
        (values[index] >= 0)
      );
    }))
  );

  ASSERT_FALSE(
    ctx.array
      .every(OCCA_FUNCTION([](const int &value) -> bool {
        return value >= 1;
      }))
  );

  ASSERT_FALSE(
    ctx.array
      .every(OCCA_FUNCTION([](const int &value, const int index) -> bool {
        return value >= 1;
      }))
  );

  ASSERT_FALSE(
    ctx.array
    .every(OCCA_FUNCTION([](const int &value, const int index, const int *values) -> bool {
      return (
        (value == values[index]) &&
        (values[index] >= 1)
      );
    }))
  );
}

void testSome(occa::device device) {
  context ctx(device);

  ASSERT_TRUE(
    ctx.array
      .some(OCCA_FUNCTION([](const int &value) -> bool {
        return value <= 0;
      }))
  );

  ASSERT_TRUE(
    ctx.array
      .some(OCCA_FUNCTION([](const int &value, const int index) -> bool {
        return value <= 0;
      }))
  );

  ASSERT_TRUE(
    ctx.array
    .some(OCCA_FUNCTION([](const int &value, const int index, const int *values) -> bool {
      return (
        (value == values[index]) &&
        (values[index] <= 0)
      );
    }))
  );

  ASSERT_FALSE(
    ctx.array
      .some(OCCA_FUNCTION([](const int &value) -> bool {
        return value < 0;
      }))
  );

  ASSERT_FALSE(
    ctx.array
      .some(OCCA_FUNCTION([](const int &value, const int index) -> bool {
        return value < 0;
      }))
  );

  ASSERT_FALSE(
    ctx.array
    .some(OCCA_FUNCTION([](const int &value, const int index, const int *values) -> bool {
      return (
        (value == values[index]) &&
        (values[index] < 0)
      );
    }))
  );
}

void testFilter(occa::device device) {
#if 0
  context ctx(device);

  occa::array<int> filteredArray;

  filteredArray = (
    ctx.array
    .filter(OCCA_FUNCTION([](const int &value) -> bool {
      return value >= 5;
    }))
  );

  ASSERT_EQ(ctx.length, (int) ctx.array.length());
  ASSERT_EQ(ctx.minValue, ctx.array.min());
  ASSERT_EQ(ctx.maxValue, ctx.array.max());

  ASSERT_EQ(5, (int) filteredArray.length());
  ASSERT_EQ(5, filteredArray.min());
  ASSERT_EQ(ctx.maxValue, filteredArray.max());

  filteredArray = (
    ctx.array
    .filter(OCCA_FUNCTION([](const int &value, const int index) -> bool {
      return index >= 5;
    }))
  );

  ASSERT_EQ(ctx.length, (int) ctx.array.length());
  ASSERT_EQ(ctx.minValue, ctx.array.min());
  ASSERT_EQ(ctx.maxValue, ctx.array.max());

  ASSERT_EQ(5, (int) filteredArray.length());
  ASSERT_EQ(5, filteredArray.min());
  ASSERT_EQ(ctx.maxValue, filteredArray.max());

  filteredArray = (
    ctx.array
    .filter(OCCA_FUNCTION([](const int &value, const int index, const int *values) -> bool {
      return values[index] >= 5;
    }))
  );

  ASSERT_EQ(ctx.length, (int) ctx.array.length());
  ASSERT_EQ(ctx.minValue, ctx.array.min());
  ASSERT_EQ(ctx.maxValue, ctx.array.max());

  ASSERT_EQ(5, (int) filteredArray.length());
  ASSERT_EQ(5, filteredArray.min());
  ASSERT_EQ(ctx.maxValue, filteredArray.max());
#endif
}

void testFindIndex(occa::device device) {
  context ctx(device);

  ASSERT_EQ(
    5, (
      ctx.array
      .findIndex(OCCA_FUNCTION([](const int &value) -> bool {
        return value == 5;
      }))
    )
  );

  ASSERT_EQ(
    6, (
      ctx.array
      .findIndex(OCCA_FUNCTION([](const int &value, const int index) -> bool {
        return index == 6;
      }))
    )
  );

  ASSERT_EQ(
    7, (
      ctx.array
      .findIndex(OCCA_FUNCTION([](const int &value, const int index, const int *values) -> bool {
        return values[index] == 7;
      }))
    )
  );

  ASSERT_EQ(
    -1, (
      ctx.array
      .findIndex(OCCA_FUNCTION([](const int &value) -> bool {
        return value == -1;
      }))
    )
  );

  ASSERT_EQ(
    -1, (
      ctx.array
      .findIndex(OCCA_FUNCTION([](const int &value, const int index) -> bool {
        return index == -1;
      }))
    )
  );

  ASSERT_EQ(
    -1, (
      ctx.array
      .findIndex(OCCA_FUNCTION([](const int &value, const int index, const int *values) -> bool {
        return values[index] == -1;
      }))
    )
  );
}

void testForEach(occa::device device) {
  context ctx(device);

  ctx.array
    .forEach(OCCA_FUNCTION([](const int &value) -> void {
      // Do nothing
    }));

  ctx.array
    .forEach(OCCA_FUNCTION([](const int &value, const int index) -> void {
      // Do nothing
    }));

  ctx.array
    .forEach(OCCA_FUNCTION([](const int &value, const int index, const int *values) -> void {
      // Do nothing
    }));
}

void testMap(occa::device device) {
  context ctx(device);

  occa::array<float> floatArray = (
    ctx.array
    .map<float>(OCCA_FUNCTION([](const int &value) -> float {
      return value / 2.0;
    }))
  );

  ASSERT_EQ(ctx.length, (int) ctx.array.length());
  ASSERT_EQ(ctx.minValue, ctx.array.min());
  ASSERT_EQ(ctx.maxValue, ctx.array.max());

  ASSERT_EQ(ctx.length, (int) floatArray.length());
  ASSERT_EQ((float) 0.0, floatArray.min());
  ASSERT_EQ((float) 4.5, floatArray.max());

  occa::array<double> doubleArray = (
    ctx.array
    .map(OCCA_FUNCTION([](const int &value, const int index) -> double {
      return 1.0 + (value + index);
    }))
  );

  ASSERT_EQ(ctx.length, (int) ctx.array.length());
  ASSERT_EQ(ctx.minValue, ctx.array.min());
  ASSERT_EQ(ctx.maxValue, ctx.array.max());

  ASSERT_EQ(ctx.length, (int) doubleArray.length());
  ASSERT_EQ((double) 1.0, doubleArray.min());
  ASSERT_EQ((double) 19.0, doubleArray.max());

  occa::array<char> charArray = (
    ctx.array
    .map(OCCA_FUNCTION([](const int &value, const int index, const int *values) -> char {
      return 'a' + (char) values[index];
    }))
  );

  ASSERT_EQ(ctx.length, (int) ctx.array.length());
  ASSERT_EQ(ctx.minValue, ctx.array.min());
  ASSERT_EQ(ctx.maxValue, ctx.array.max());

  ASSERT_EQ(ctx.length, (int) charArray.length());
  ASSERT_EQ('a', charArray.min());
  ASSERT_EQ('j', charArray.max());
}

void testMapTo(occa::device device) {
  context ctx(device);

  ctx.array
    .mapTo(ctx.array, OCCA_FUNCTION([](const int &value) -> int {
      return value / 2;
    }));

  ASSERT_EQ(ctx.length, (int) ctx.array.length());
  ASSERT_EQ(ctx.minValue, ctx.array.min());
  ASSERT_EQ(ctx.maxValue / 2, ctx.array.max());

  ctx.array
    .mapTo(ctx.array, OCCA_FUNCTION([](const int &value, const int index) -> int {
      return index / 5;
    }));

  ASSERT_EQ(ctx.length, (int) ctx.array.length());
  ASSERT_EQ(ctx.minValue, ctx.array.min());
  ASSERT_EQ(1, ctx.array.max());

  ctx.array
    .mapTo(ctx.array, OCCA_FUNCTION([](const int &value, const int index, const int *values) -> int {
      return index + values[index];
    }));

  ASSERT_EQ(ctx.length, (int) ctx.array.length());
  ASSERT_EQ(ctx.minValue, ctx.array.min());
  ASSERT_EQ(10, ctx.array.max());
}

void testReduce(occa::device device) {
  context ctx(device);

  int addReduction = 0;
  float subtractReduction = 0;
  long multiplyReduction = 0;
  int boolOrReduction = 0;
  int boolAndReduction = 0;
  int minReduction = 0;
  int maxReduction = 0;

  for (int i = 0; i < ctx.length; ++i) {
    const int value = ctx.values[i];
    addReduction += value;
    subtractReduction -= value;
    multiplyReduction *= value;
    boolOrReduction = boolOrReduction || value;
    boolAndReduction = boolAndReduction && value;
    minReduction = std::min(minReduction, value);
    maxReduction = std::max(maxReduction, value);
  }

  ASSERT_EQ(
    addReduction,
    ctx.array
    .reduce<int>(
      occa::reductionType::sum,
      OCCA_FUNCTION([](const int &acc, const int &value) -> int {
        return acc + value;
      })
    )
  );

  ASSERT_EQ(
    subtractReduction,
    ctx.array
    .reduce<float>(
      occa::reductionType::sum,
      OCCA_FUNCTION([](const float &acc, const int &value) -> float {
        return acc - value;
      })
    )
  );

  ASSERT_EQ(
    multiplyReduction,
    ctx.array
    .reduce<int>(
      occa::reductionType::multiply,
      OCCA_FUNCTION([](const int &acc, const int &value) -> int {
        return acc * value;
      })
    )
  );

  ASSERT_EQ(
    boolOrReduction,
    ctx.array
    .reduce<bool>(
      occa::reductionType::boolOr,
      OCCA_FUNCTION([](const bool &acc, const int &value) -> bool {
        return acc || value;
      })
    )
  );

  ASSERT_EQ(
    boolAndReduction,
    ctx.array
    .reduce<bool>(
      occa::reductionType::boolAnd,
      OCCA_FUNCTION([](const bool &acc, const int &value) -> bool {
        return acc && value;
      })
    )
  );

  ASSERT_EQ(
    minReduction,
    ctx.array
    .reduce<int>(
      occa::reductionType::min,
      OCCA_FUNCTION([](const int &acc, const int &value) -> int {
        return acc < value ? acc : value;
      })
    )
  );

  ASSERT_EQ(
    maxReduction,
    ctx.array
    .reduce<int>(
      occa::reductionType::max,
      OCCA_FUNCTION([](const int &acc, const int &value) -> int {
        return acc > value ? acc : value;
      })
    )
  );
}

void testSlice(occa::device device) {
  context ctx(device);

  occa::array<int> slice;

  slice = ctx.array.slice(0, 2);
  ASSERT_EQ(ctx.length, (int) ctx.array.length());
  ASSERT_EQ(ctx.minValue, ctx.array.min());
  ASSERT_EQ(ctx.maxValue, ctx.array.max());

  ASSERT_EQ(2, (int) slice.length());
  ASSERT_EQ(ctx.minValue, slice.min());
  ASSERT_EQ(1, slice.max());

  slice = ctx.array.slice(5);
  ASSERT_EQ(ctx.length, (int) ctx.array.length());
  ASSERT_EQ(ctx.minValue, ctx.array.min());
  ASSERT_EQ(ctx.maxValue, ctx.array.max());

  ASSERT_EQ(5, (int) slice.length());
  ASSERT_EQ(5, slice.min());
  ASSERT_EQ(ctx.maxValue, slice.max());

  slice = ctx.array.slice(8, 1);
  ASSERT_EQ(ctx.length, (int) ctx.array.length());
  ASSERT_EQ(ctx.minValue, ctx.array.min());
  ASSERT_EQ(ctx.maxValue, ctx.array.max());

  ASSERT_EQ(1, (int) slice.length());
  ASSERT_EQ(8, slice.min());
  ASSERT_EQ(8, slice.max());
}

void testConcat(occa::device device) {
  context ctx(device);

  occa::array<int> two   = ctx.array.concat(ctx.array);
  occa::array<int> three = two.concat(ctx.array);

  ASSERT_EQ(ctx.length, (int) ctx.array.length());
  ASSERT_EQ(ctx.minValue, ctx.array.min());
  ASSERT_EQ(ctx.maxValue, ctx.array.max());

  ASSERT_EQ(2 * ctx.length, (int) two.length());
  ASSERT_EQ(ctx.minValue, two.min());
  ASSERT_EQ(ctx.maxValue, two.max());

  ASSERT_EQ(3 * ctx.length, (int) three.length());
  ASSERT_EQ(ctx.minValue, three.min());
  ASSERT_EQ(ctx.maxValue, three.max());

  int *h_two = new int[2 * ctx.length];
  int *h_three = new int[3 * ctx.length];

  for (int i = 0; i < (2 * ctx.length); ++i) {
    h_two[i] = -1;
  }

  for (int i = 0; i < (3 * ctx.length); ++i) {
    h_three[i] = -1;
  }

  two.copyTo(h_two);
  three.copyTo(h_three);

  for (int i = 0; i < (2 * ctx.length); ++i) {
    ASSERT_EQ(i % ctx.length, h_two[i]);
  }

  for (int i = 0; i < (3 * ctx.length); ++i) {
    ASSERT_EQ(i % ctx.length, h_three[i]);
  }

  delete [] h_two;
  delete [] h_three;
}

void testFill(occa::device device) {
  context ctx(device);

  for (int i = 0; i < ctx.length; ++i) {
    ctx.values[i] = 0;
  }

  ctx.array.fill(5);
  ctx.array.copyTo(ctx.values);

  for (int i = 0; i < ctx.length; ++i) {
    ASSERT_EQ(5, ctx.values[i]);
  }

  ctx.array.fill(6);
  ctx.array.copyTo(ctx.values);

  for (int i = 0; i < ctx.length; ++i) {
    ASSERT_EQ(6, ctx.values[i]);
  }
}

void testIncludes(occa::device device) {
  context ctx(device);

  ASSERT_TRUE(
    ctx.array.includes(5)
  );

  ASSERT_TRUE(
    ctx.array.includes(6)
  );

  ASSERT_TRUE(
    ctx.array.includes(7)
  );

  ASSERT_FALSE(
    ctx.array.includes(-1)
  );
}

void testIndexOf(occa::device device) {
  context ctx(device);

  for (int i = 0; i < ctx.length; ++i) {
    ctx.values[i] = i % 2;
  }
  ctx.array.copyFrom(ctx.values);

  ASSERT_EQ(
    0,
    ctx.array.indexOf(0)
  );

  ASSERT_EQ(
    1,
    ctx.array.indexOf(1)
  );

  ASSERT_EQ(
    -1,
    ctx.array.indexOf(2)
  );
}

void testLastIndexOf(occa::device device) {
  context ctx(device);

  for (int i = 0; i < ctx.length; ++i) {
    ctx.values[i] = i % 2;
  }
  ctx.array.copyFrom(ctx.values);

  ASSERT_EQ(
    ctx.length - 2,
    ctx.array.lastIndexOf(0)
  );

  ASSERT_EQ(
    ctx.length - 1,
    ctx.array.lastIndexOf(1)
  );

  ASSERT_EQ(
    -1,
    ctx.array.lastIndexOf(2)
  );
}

void testCast(occa::device device) {
  context ctx(device);

  occa::array<int> intArray = ctx.array.cast<int>();
  occa::array<char> charArray = ctx.array.cast<char>();

  ASSERT_EQ(ctx.length, (int) ctx.array.length());
  ASSERT_EQ(ctx.minValue, ctx.array.min());
  ASSERT_EQ(ctx.maxValue, ctx.array.max());

  ASSERT_EQ(ctx.length, (int) intArray.length());
  ASSERT_EQ(ctx.minValue, intArray.min());
  ASSERT_EQ(ctx.maxValue, intArray.max());

  ASSERT_EQ(ctx.length, (int) charArray.length());
  ASSERT_EQ('a', 'a' + charArray.min());
  ASSERT_EQ('j', 'a' + charArray.max());
}

void testReverse(occa::device device) {
  context ctx(device);

  occa::array<int> reversedArray = (
    ctx.array.reverse()
  );

  ASSERT_EQ(ctx.length, (int) ctx.array.length());
  ASSERT_EQ(ctx.minValue, ctx.array.min());
  ASSERT_EQ(ctx.maxValue, ctx.array.max());

  ASSERT_EQ(ctx.length, (int) reversedArray.length());
  ASSERT_EQ(ctx.minValue, reversedArray.min());
  ASSERT_EQ(ctx.maxValue, reversedArray.max());

  int *reversedValues = new int[ctx.length];

  reversedArray.copyTo(reversedValues);

  for (int i = 0; i < ctx.length; ++i) {
    ASSERT_EQ(
      ctx.values[i],
      reversedValues[ctx.length - i - 1]
    );
  }
}

void testShiftLeft(occa::device device) {
  context ctx(device);

  occa::array<int> shiftedArray = (
    ctx.array.shiftLeft(3, -1)
  );

  ASSERT_EQ(ctx.length, (int) ctx.array.length());
  ASSERT_EQ(ctx.minValue, ctx.array.min());
  ASSERT_EQ(ctx.maxValue, ctx.array.max());

  int *shiftedValues = new int[ctx.length];
  shiftedArray.copyTo(shiftedValues);

  for (int i = 0; i < (ctx.length - 3); ++i) {
    ASSERT_EQ(
      ctx.values[i + 3],
      shiftedValues[i]
    );
  }

  for (int i = (ctx.length - 3); i < ctx.length; ++i) {
    ASSERT_EQ(-1, shiftedValues[i]);
  }
}

void testShiftRight(occa::device device) {
  context ctx(device);

  occa::array<int> shiftedArray = (
    ctx.array.shiftRight(3, -1)
  );

  ASSERT_EQ(ctx.length, (int) ctx.array.length());
  ASSERT_EQ(ctx.minValue, ctx.array.min());
  ASSERT_EQ(ctx.maxValue, ctx.array.max());

  int *shiftedValues = new int[ctx.length];
  shiftedArray.copyTo(shiftedValues);

  for (int i = 0; i < 3; ++i) {
    ASSERT_EQ(-1, shiftedValues[i]);
  }

  for (int i = 3; i < ctx.length; ++i) {
    ASSERT_EQ(
      ctx.values[i - 3],
      shiftedValues[i]
    );
  }
}

void testMax(occa::device device) {
  context ctx(device);

  ASSERT_EQ(ctx.maxValue, ctx.array.max());
}

void testMin(occa::device device) {
  context ctx(device);

  ASSERT_EQ(ctx.minValue, ctx.array.min());
}

void testDotProduct(occa::device device) {
  context ctx(device);

  int dotProduct = 0;
  for (int i = 0; i < ctx.length; ++i) {
    dotProduct += (ctx.values[i] * ctx.values[i]);
  }

  ASSERT_EQ(dotProduct,
            ctx.array.dotProduct(ctx.array));
}

void testClamp(occa::device device) {
  context ctx(device);

  occa::array<int> clampedArray = ctx.array.clamp(4, 7);

  ASSERT_EQ(4, clampedArray.min());
  ASSERT_EQ(7, clampedArray.max());

  clampedArray = ctx.array.clampMin(4);

  ASSERT_EQ(4, clampedArray.min());
  ASSERT_EQ(9, clampedArray.max());

  clampedArray = ctx.array.clampMax(7);

  ASSERT_EQ(0, clampedArray.min());
  ASSERT_EQ(7, clampedArray.max());
}
