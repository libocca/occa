#include <iostream>
#include <occa.hpp>
#include <vector>
#include "constants.h"

std::vector<float> buildData(std::size_t size,
                             float initialValue,
                             float fluctuation)
{
    std::vector<float> buffer(size);
    float currentValue = initialValue;
    float longIncrement = 1.0f;
    float fluctuationIncrement = fluctuation;
    for(std::size_t i = 0; i < buffer.size(); ++i) {
        buffer[i] = currentValue;
        fluctuationIncrement = -fluctuationIncrement;
        if(i % WINDOW_SIZE == 0) {
            longIncrement = -longIncrement;
        }
        currentValue += longIncrement + fluctuationIncrement;
    }
    return buffer;
}

std::vector<float> goldMovingAverage(const std::vector<float> &hostVector) {
    std::vector<float> result(hostVector.size() - WINDOW_SIZE);
    for(std::size_t i = 0; i < result.size(); ++i) {
        float value = 0.0f;
        for(std::size_t j = 0; j < WINDOW_SIZE; ++j) {
            value += hostVector[i + j];
        }
        result[i] = value / WINDOW_SIZE;
    }
    return result;
}

bool starts_with(const std::string &str, const std::string &substring) {
    return str.rfind(substring, 0) == 0;
}

occa::json getDeviceOptions(int argc, const char **argv) {
    for(int i  = 0; i < argc; ++i) {
        std::string argument(argv[i]);
        if((starts_with(argument,"-d") || starts_with(argument, "--device")) && i + 1 < argc)
        {
            std::string value(argv[i + 1]);
            return occa::json::parse(value);
        }
    }
    return occa::json::parse("{mode: 'Serial'}");
}

int main(int argc, const char **argv) {

  occa::json deviceOpts = getDeviceOptions(argc, argv);
  auto inputHostBuffer = buildData(THREADS_PER_BLOCK * WINDOW_SIZE + WINDOW_SIZE, 10.0f, 4.0f);
  std::vector<float> outputHostBuffer(inputHostBuffer.size() - WINDOW_SIZE);

  occa::device device(deviceOpts);
  occa::memory deviceInput = device.malloc<float>(inputHostBuffer.size());
  occa::memory deviceOutput = device.malloc<float>(outputHostBuffer.size());

  occa::json buildProps({
      {"transpiler-version", 3}
  });

  occa::kernel movingAverageKernel = device.buildKernel("movingAverage.okl", "movingAverage32f", buildProps);

  deviceInput.copyFrom(inputHostBuffer.data(), inputHostBuffer.size());

  movingAverageKernel(deviceInput,
                      static_cast<int>(inputHostBuffer.size()),
                      deviceOutput,
                      static_cast<int>(deviceOutput.size()));

  // Copy result to the host
  deviceOutput.copyTo(&outputHostBuffer[0], outputHostBuffer.size());

  auto goldValue = goldMovingAverage(inputHostBuffer);

  constexpr const float EPSILON = 0.001f;
  for(std::size_t i = 0; i < outputHostBuffer.size(); ++i) {
        bool isValid = std::abs(goldValue[i] - outputHostBuffer[i]) < EPSILON;
        if(!isValid) {
            std::cout << "Comparison with gold values has failed" << std::endl;
            return 1;
        }
  }
  std::cout << "Comparison with gold has passed" << std::endl;
  std::cout << "Moving average finished" << std::endl;

  return 0;
}
