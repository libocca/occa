#include <occa.hpp>
#include <occa/internal/utils/testing.hpp>

// TODO: Handle occa:: types in OKL
using int2 = occa::int2;
using int3 = occa::int3;

void testOuterForLoops(occa::device device);
void testFullForLoops(occa::device device);

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
    testOuterForLoops(device);
    testFullForLoops(device);
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
    .run(scope, OCCA_FUNCTION({}, [=](int outerIndex) -> void {
      OKL("@inner")
      for (int i = 0; i < 2; ++i) {
        const int globalIndex = i + (2 * outerIndex);
        output[globalIndex] = -globalIndex;
      }
    }));

  ASSERT_EQ(0, output[0]);
  ASSERT_EQ((2 * length) - 1,
            output[(2 * length) - 1]);
  ASSERT_EQ(-1,
            output[2 * length]);

  occa::forLoop()
    .outer(length, occa::range(length))
    .run(scope, OCCA_FUNCTION({}, [=](int2 outerIndex) -> void {
      OKL("@inner")
      for (int i = 0; i < 2; ++i) {
        const int globalIndex = (
          i + (2 * (outerIndex.y + length * outerIndex.x))
        );
        output[globalIndex] = -globalIndex;
      }
    }));

  ASSERT_EQ(0, output[0]);
  ASSERT_EQ((2 * length) - 1,
            output[2 * length * length - 1]);
  ASSERT_EQ(-1,
            output[2 * length * length]);

  occa::forLoop()
    .outer(length, occa::range(length), indexArray)
    .run(scope, OCCA_FUNCTION({}, [=](int3 outerIndex) -> void {
      OKL("@inner")
      for (int i = 0; i < 2; ++i) {
        const int globalIndex = (
          i + (2 * (outerIndex.z + length * (outerIndex.y + length * outerIndex.x)))
        );
        output[globalIndex] = globalIndex;
      }
    }));

  ASSERT_EQ(0, output[0]);
  ASSERT_EQ((2 * length * length * length) - 1,
            output[(2 * length * length * length) - 1]);
}

void testFullForLoops(occa::device device) {
}
