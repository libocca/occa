#include <occa.hpp>
#include <occa/internal/utils/testing.hpp>

// TODO: Handle occa:: types in OKL
using int2 = occa::int2;
using int3 = occa::int3;

void testOuterForLoops(occa::device device);
void testFullForLoops(occa::device device);
void testTileForLoops(occa::device device);

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

  for (auto &device : devices) {
    std::cout << "Testing mode: " << device.mode() << '\n';
    testOuterForLoops(device);
    testFullForLoops(device);
    testTileForLoops(device);
  }
}

void testOuterForLoops(occa::device device) {
  const int length = 10;
  occa::array<int> indexArray = occa::range(length).toArray();

  occa::array<float> output(length * length * length * 2);
  output = output.fill(-1);

  occa::scope scope({
    {"output", output}
  }, {
    {"defines/length", length}
  });

  occa::forLoop()
    .outer(length)
    .run(OCCA_FUNCTION(scope, [=](const int outerIndex) -> void {
      OKL("@inner");
      for (int i = 0; i < 2; ++i) {
        const int globalIndex = i + (2 * outerIndex);
        output[globalIndex] = globalIndex;
      }
    }));

  ASSERT_EQ((float) 0, output[0]);
  ASSERT_EQ((float) (2 * length) - 1,
            output[(2 * length) - 1]);
  ASSERT_EQ((float) -1,
            output[2 * length]);

  occa::forLoop()
    .outer(length, occa::range(length))
    .run(OCCA_FUNCTION(scope, [=](const int2 outerIndex) -> void {
      OKL("@inner");
      for (int i = 0; i < 2; ++i) {
        const int globalIndex = (
          i + (2 * (outerIndex.y + length * outerIndex.x))
        );
        output[globalIndex] = -globalIndex;
      }
    }));

  ASSERT_EQ((float) 0, output[0]);
  ASSERT_EQ((float) -((2 * length * length) - 1),
            output[2 * length * length - 1]);
  ASSERT_EQ((float) -1,
            output[2 * length * length]);

  occa::forLoop()
    .outer(length, occa::range(length), indexArray)
    .run(OCCA_FUNCTION(scope, [=](const int3 outerIndex) -> void {
      OKL("@inner");
      for (int i = 0; i < 2; ++i) {
        const int globalIndex = (
          i + (2 * (outerIndex.z + length * (outerIndex.y + length * outerIndex.x)))
        );
        output[globalIndex] = globalIndex;
      }
    }));

  ASSERT_EQ((float) 0, output[0]);
  ASSERT_EQ((float) (2 * length * length * length) - 1,
            output[(2 * length * length * length) - 1]);

  occa::forLoop()
    .outer(length)
    .run(OCCA_FUNCTION(scope, [=](const int outerIndex) -> void {
      OKL("@shared"); int array[2];

      OKL("@inner");
      for (int i = 0; i < 2; ++i) {
        array[i] = i;
      }

      OKL("@inner");
      for (int i = 0; i < 2; ++i) {
        output[i] = array[1 - i];
      }
    }));

  ASSERT_EQ((float) 1, output[0]);
  ASSERT_EQ((float) 0, output[1]);
}

void testFullForLoops(occa::device device) {
  const int length = 10;
  occa::array<int> indexArray = occa::range(length).toArray();

  occa::array<float> output(length * length * length * 2);
  output = output.fill(-1);

  occa::scope scope({
    {"output", output}
  }, {
    {"defines/length", length}
  });

  occa::forLoop()
    .outer(2)
    .inner(length)
    .run(OCCA_FUNCTION(scope, [=](const int outerIndex, const int innerIndex) -> void {
      const int globalIndex = outerIndex + (2 * innerIndex);
      output[globalIndex] = globalIndex;
    }));

  ASSERT_EQ((float) 0, output[0]);
  ASSERT_EQ((float) (2 * length) - 1,
            output[(2 * length) - 1]);
  ASSERT_EQ((float) -1,
            output[2 * length]);

  occa::forLoop()
    .outer(2)
    .inner(length, occa::range(length))
    .run(OCCA_FUNCTION(scope, [=](const int outerIndex, const int2 innerIndex) -> void {
      const int globalIndex = (
        outerIndex + (2 * (innerIndex.y + length * innerIndex.x))
      );
      output[globalIndex] = -globalIndex;
    }));

  ASSERT_EQ((float) 0, output[0]);
  ASSERT_EQ((float) -((2 * length * length) - 1),
            output[2 * length * length - 1]);
  ASSERT_EQ((float) -1,
            output[2 * length * length]);

  occa::forLoop()
    .outer(2)
    .inner(length, occa::range(length), indexArray)
    .run(OCCA_FUNCTION(scope, [=](const int outerIndex, const int3 innerIndex) -> void {
      const int globalIndex = (
        outerIndex + (2 * (innerIndex.z + length * (innerIndex.y + length * innerIndex.x)))
      );
      output[globalIndex] = globalIndex;
    }));

  ASSERT_EQ((float) 0, output[0]);
  ASSERT_EQ((float) (2 * length * length * length) - 1,
            output[(2 * length * length * length) - 1]);
}

void testTileForLoops(occa::device device) {
  const int length = 10;
  occa::array<int> indexArray = occa::range(length).toArray();

  occa::array<float> output(length * length * length);
  output = output.fill(-1);

  occa::scope scope({
    {"output", output}
  }, {
    {"defines/length", length}
  });

  occa::forLoop()
    .tile({length, 2})
    .run(OCCA_FUNCTION(scope, [=](const int index) -> void {
      output[index] = index;
    }));

  ASSERT_EQ((float) 0, output[0]);
  ASSERT_EQ((float) length - 1,
            output[length - 1]);
  ASSERT_EQ((float) -1,
            output[length]);

  occa::forLoop()
    .tile({length, 2}, {occa::range(length), 2})
    .run(OCCA_FUNCTION(scope, [=](const int2 index) -> void {
      const int globalIndex = (
        index.x + (length * index.y)
      );
      output[globalIndex] = -globalIndex;
    }));

  ASSERT_EQ((float) 0, output[0]);
  ASSERT_EQ((float) -((length * length) - 1),
            output[(length * length) - 1]);
  ASSERT_EQ((float) -1,
            output[length * length]);

  occa::forLoop()
    .tile(
      {length, 2},
      {occa::range(length), 2},
      {indexArray, 2}
    )
    .run(OCCA_FUNCTION(scope, [=](const int3 index) -> void {
      const int globalIndex = (
        index.x + (length * (index.y + length * index.z))
      );
      output[globalIndex] = globalIndex;
    }));

  ASSERT_EQ((float) 0, output[0]);
  ASSERT_EQ((float) (length * length * length) - 1,
            output[(length * length * length) - 1]);
}
