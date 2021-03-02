#ifndef OCCA_FUNCTIONAL_TYPELESSARRAY_HEADER
#define OCCA_FUNCTIONAL_TYPELESSARRAY_HEADER

#include <occa/defines/okl.hpp>
#include <occa/dtype.hpp>
#include <occa/core.hpp>
#include <occa/functional/function.hpp>
#include <occa/functional/utils.hpp>
#include <occa/experimental/kernelBuilder.hpp>

namespace occa {
  class kernelArg;

  class typelessArray {
  protected:
    mutable occa::device device_;
    dtype_t dtype_;

    int tileSize;
    int tileIterations;

    // Buffer memory
    mutable occa::memory returnMemory;

    template <class ReturnType>
    void setupReturnMemory(const ReturnType &value) const {
      setupReturnMemoryArray<ReturnType>(1);
      returnMemory.copyFrom(&value, sizeof(ReturnType));
    }

    template <class ReturnType>
    void setupReturnMemoryArray(const int size) const {
      size_t bytes = sizeof(ReturnType) * size;
      if (bytes > returnMemory.size()) {
        returnMemory = device_.template malloc<ReturnType>(size);
      }
      returnMemory.setDtype(dtype::get<ReturnType>());
    }

    template <class ReturnType>
    void setReturnValue(ReturnType &value) const {
      size_t bytes = sizeof(ReturnType);
      returnMemory.copyTo(&value, bytes);
    }

    virtual occa::scope getMapArrayScopeOverrides() const {
      return occa::scope();
    }

    virtual occa::scope getReduceArrayScopeOverrides() const {
      return occa::scope();
    }

    occa::scope getMapArrayScope(const baseFunction &fn) const {
      const int arrayLength = (int) length();

      const int safeTileSize = std::min(
        std::max(1, tileSize),
        arrayLength
      );
      const int safeTileIterations = std::min(
        std::max(1, tileIterations),
        (arrayLength + safeTileSize - 1) / safeTileSize
      );

      std::string tileForLoop;
      std::string parallelForLoop;
      buildMapTiledForLoops(tileForLoop, parallelForLoop);

      occa::scope baseScope({
        {"occa_array_length", arrayLength},
        {"occa_array_return", returnMemory}
      }, {
        {"defines/T", dtype_.name()},
        {"defines/OCCA_ARRAY_TILE_SIZE", safeTileSize},
        {"defines/OCCA_ARRAY_TILE_ITERATIONS", safeTileIterations},
        {"defines/OCCA_ARRAY_FUNCTION(VALUE, INDEX, VALUES_PTR)", buildMapFunctionCall(fn)},
        {"defines/OCCA_ARRAY_TILE_FOR_LOOP", tileForLoop},
        {"defines/OCCA_ARRAY_TILE_PARALLEL_FOR_LOOP", parallelForLoop},
        {"functions/occa_array_function", fn}
      });

      baseScope.device = device_;

      return (
        baseScope
        + getMapArrayScopeOverrides()
        + fn.scope
      );
    }

    template <class T2>
    occa::scope getCpuReduceArrayScope(reductionType type,
                                       const T2 &localInit,
                                       const bool useLocalInit,
                                       const baseFunction &fn) const {
      const int arrayLength = (int) length();
      const int ompLoopSize = 128;

      setupReturnMemoryArray<T2>(ompLoopSize);

      occa::json props({
        {"defines/T", dtype_.name()},
        {"defines/T2", dtype::get<T2>().name()},
        {"defines/OCCA_ARRAY_OMP_LOOP_SIZE", 128},
        {"defines/OCCA_ARRAY_FUNCTION(ACC, VALUE, INDEX, VALUES_PTR)", buildReduceFunctionCall(fn)},
        {"defines/OCCA_ARRAY_LOCAL_REDUCTION(LEFT_VALUE, RIGHT_VALUE)", buildLocalReductionOperation(type)},
        {"functions/occa_array_function", fn}
      });

      if (useLocalInit) {
        props["defines/OCCA_ARRAY_REDUCTION_INIT_VALUE"] = localInit;
      } else {
        props["defines/OCCA_ARRAY_REDUCTION_INIT_VALUE"] = buildReductionInitValue(type);
      }

      occa::scope baseScope({
        {"occa_array_length", arrayLength},
        {"occa_array_return", returnMemory}
      }, props);

      baseScope.device = device_;

      return (
        baseScope
        + getReduceArrayScopeOverrides()
        + fn.scope
      );
    }

    template <class T2>
    occa::scope getGpuReduceArrayScope(reductionType type,
                                       const T2 &localInit,
                                       const bool useLocalInit,
                                       const baseFunction &fn) const {
      const int arrayLength = (int) length();

      // Default and limit to 1024 if not set
      int unsafeTileSize = (
        tileSize <= 0
        ? 1024
        : std::min(1024, tileSize)
      );

      // Limit it to the array length
      unsafeTileSize = std::min(unsafeTileSize, arrayLength);

      // Make sure it's a power of 2
      int safeTileSize = 1024;
      while ((safeTileSize > 1) && ((safeTileSize >> 1) > unsafeTileSize)) {
        safeTileSize >>= 1;
      }

      // Default to 16 reductions locally
      const int defaultTileIterations = (
        tileIterations <= 0
        ? 16
        : tileIterations
      );
      const int safeTileIterations = std::min(
        defaultTileIterations,
        (arrayLength + safeTileSize - 1) / safeTileSize
      );

      const int localReductionSize = safeTileSize * safeTileIterations;
      const int localReductionCount = (arrayLength + localReductionSize - 1) / localReductionSize;

      setupReturnMemoryArray<T2>(localReductionCount);

      occa::json props({
        {"defines/T", dtype_.name()},
        {"defines/T2", dtype::get<T2>().name()},
        {"defines/OCCA_ARRAY_TILE_SIZE", safeTileSize},
        {"defines/OCCA_ARRAY_TILE_ITERATIONS", safeTileIterations},
        {"defines/OCCA_ARRAY_FUNCTION(ACC, VALUE, INDEX, VALUES_PTR)", buildReduceFunctionCall(fn)},
        {"defines/OCCA_ARRAY_LOCAL_REDUCTION(LEFT_VALUE, RIGHT_VALUE)", buildLocalReductionOperation(type)},
        {"defines/OCCA_ARRAY_SHARED_REDUCTION(BOUNDS)",
         "for (int i = 0; i < OCCA_ARRAY_TILE_SIZE; ++i; @inner) {"
         "  if (i < BOUNDS) {"
         "    const T2 leftValue = tileAcc[i];"
         "    const T2 rightValue = tileAcc[i + BOUNDS];"
         "    tileAcc[i] = OCCA_ARRAY_LOCAL_REDUCTION(leftValue, rightValue);"
         "  }"
         "}"},
        {"functions/occa_array_function", fn}
      });

      if (useLocalInit) {
        props["defines/OCCA_ARRAY_REDUCTION_INIT_VALUE"] = localInit;
      } else {
        props["defines/OCCA_ARRAY_REDUCTION_INIT_VALUE"] = buildReductionInitValue(type);
      }

      occa::scope baseScope({
        {"occa_array_length", arrayLength},
        {"occa_array_return", returnMemory}
      }, props);

      baseScope.device = device_;

      return (
        baseScope
        + getReduceArrayScopeOverrides()
        + fn.scope
      );
    }

    std::string buildMapFunctionCall(const baseFunction &fn) const {
      return buildFunctionCall(fn, true);
    }

    std::string buildReduceFunctionCall(const baseFunction &fn) const {
      return buildFunctionCall(fn, false);
    }

    std::string buildFunctionCall(const baseFunction &fn,
                                  const bool forMapFunction) const {
      strVector argumentValues;
      if (forMapFunction) {
        argumentValues = {"VALUE", "INDEX", "VALUES_PTR"};
      } else {
        argumentValues = {"ACC", "VALUE", "INDEX", "VALUES_PTR"};
      }
      argumentValues.resize(fn.argumentCount());

      return fn.buildFunctionCall("occa_array_function",
                                  argumentValues);
    }

    void buildMapTiledForLoops(std::string &tileForLoop,
                               std::string &parallelForLoop) const {
      if (usingNativeCpuMode()) {
        buildCpuMapTiledForLoops(tileForLoop, parallelForLoop);
      } else {
        buildGpuMapTiledForLoops(tileForLoop, parallelForLoop);
      }
    }

    void buildCpuMapTiledForLoops(std::string &tileForLoop,
                                  std::string &parallelForLoop) const {
      tileForLoop = (
        "for (int tileIndex = 0;"
        " tileIndex < occa_array_length;"
        " tileIndex += OCCA_ARRAY_TILE_ITERATIONS;"
        " @tile(OCCA_ARRAY_TILE_SIZE * OCCA_ARRAY_TILE_ITERATIONS, @outer, @inner, check=false))"
      );

      parallelForLoop = (
        "for (int i = tileIndex;"
        " i < tileIndex + OCCA_ARRAY_TILE_ITERATIONS;"
        " ++i)"
        "  if (i < occa_array_length)"
      );
    }

    void buildGpuMapTiledForLoops(std::string &tileForLoop,
                                  std::string &parallelForLoop) const {
      tileForLoop = (
        "for (int tileIndex = 0;"
        " tileIndex < occa_array_length;"
        " tileIndex += OCCA_ARRAY_TILE_SIZE;"
        " @tile(OCCA_ARRAY_TILE_SIZE * OCCA_ARRAY_TILE_ITERATIONS, @outer, @inner, check=false))"
      );

      parallelForLoop = (
        "for (int i = tileIndex;"
        " i < tileIndex + (OCCA_ARRAY_TILE_SIZE * OCCA_ARRAY_TILE_ITERATIONS);"
        " i += OCCA_ARRAY_TILE_ITERATIONS)"
        "  if (i < occa_array_length)"
      );
    }

    virtual std::string reductionInitialValue() const = 0;

    std::string buildReductionInitValue(reductionType type) const {
      switch (type) {
        case reductionType::sum:
          return "0";
        case reductionType::multiply:
          return "1";
        case reductionType::bitOr:
          return "0";
        case reductionType::bitAnd:
          return reductionInitialValue();
        case reductionType::bitXor:
          return "0";
        case reductionType::boolOr:
          return "0";
        case reductionType::boolAnd:
          return reductionInitialValue();
        case reductionType::min:
          return reductionInitialValue();
        case reductionType::max:
          return reductionInitialValue();
        default:
          // Shouldn't get here
          return "";
      }
    }

    std::string buildLocalReductionOperation(reductionType type) const {
      switch (type) {
        case reductionType::sum:
          return "LEFT_VALUE + RIGHT_VALUE";
        case reductionType::multiply:
          return "LEFT_VALUE * RIGHT_VALUE";
        case reductionType::bitOr:
          return "LEFT_VALUE | RIGHT_VALUE";
        case reductionType::bitAnd:
          return "LEFT_VALUE & RIGHT_VALUE";
        case reductionType::bitXor:
          return "LEFT_VALUE ^ RIGHT_VALUE";
        case reductionType::boolOr:
          return "LEFT_VALUE || RIGHT_VALUE";
        case reductionType::boolAnd:
          return "LEFT_VALUE && RIGHT_VALUE";
        case reductionType::min:
          return "LEFT_VALUE < RIGHT_VALUE ? LEFT_VALUE : RIGHT_VALUE";
        case reductionType::max:
          return "LEFT_VALUE > RIGHT_VALUE ? LEFT_VALUE : RIGHT_VALUE";
        default:
          // Shouldn't get here
          return "";
      }
    }

    bool usingNativeCpuMode() const {
      const std::string &mode = device_.mode();
      return (mode == "Serial" || mode == "OpenMP");
    }

  public:
    typelessArray() :
      tileSize(-1),
      tileIterations(-1) {}

    typelessArray(const typelessArray &other) :
      device_(other.device_),
      dtype_(other.dtype_),
      tileSize(other.tileSize),
      tileIterations(other.tileIterations) {}

    typelessArray& operator = (const typelessArray &other) {
      device_ = other.device_;
      dtype_ = other.dtype_;

      tileSize = other.tileSize;
      tileIterations = other.tileIterations;

      return *this;
    }

  protected:
    typelessArray& setupTypelessArray(occa::device device__, const dtype_t &dtype__) {
      device_ = device__;
      dtype_ = dtype__;

      return *this;
    }

    typelessArray& setupTypelessArray(occa::memory mem) {
      device_ = mem.getDevice();
      dtype_ = mem.dtype();

      return *this;
    }

  public:
    void setTileSize(const int tileSize_) {
      setTileSize(tileSize_, 1);
    }

    void setTileSize(const int tileSize_,
                     const int tileIterations_) {
      if (tileSize_ > 0) {
        tileSize = tileSize_;
      }
      if (tileIterations_ > 0) {
        tileIterations = tileIterations_;
      }
    }

    //---[ Memory methods ]-------------
    occa::device getDevice() const {
      return device_;
    }

    occa::dtype_t dtype() const {
      return dtype_;
    }

    virtual udim_t length() const = 0;
    //==================================

  protected:
    //---[ Lambda methods ]-------------
    bool typelessEvery(const baseFunction &fn) const {
      bool returnValue = true;

      setupReturnMemory(returnValue);

      OCCA_JIT(getMapArrayScope(fn), (
        OCCA_ARRAY_TILE_FOR_LOOP {
          OCCA_ARRAY_TILE_PARALLEL_FOR_LOOP {
            if (!OCCA_ARRAY_FUNCTION_CALL(i)) {
              occa_array_return[0] = false;
            }
          }
        }
      ));

      setReturnValue(returnValue);

      return returnValue;
    }

    bool typelessSome(const baseFunction &fn) const {
      return typelessFindIndex(fn) >= 0;
    }

    int typelessFindIndex(const baseFunction &fn) const {
      int returnValue = -1;

      setupReturnMemory(returnValue);

      OCCA_JIT(getMapArrayScope(fn), (
        OCCA_ARRAY_TILE_FOR_LOOP {
          OCCA_ARRAY_TILE_PARALLEL_FOR_LOOP {
            if (OCCA_ARRAY_FUNCTION_CALL(i)) {
              occa_array_return[0] = i;
            }
          }
        }
      ));

      setReturnValue(returnValue);

      return returnValue;
    }

    void typelessForEach(const baseFunction &fn) const {
      OCCA_JIT(getMapArrayScope(fn), (
        OCCA_ARRAY_TILE_FOR_LOOP {
          OCCA_ARRAY_TILE_PARALLEL_FOR_LOOP {
            OCCA_ARRAY_FUNCTION_CALL(i);
          }
        }
      ));
    }

    template <class T2>
    occa::memory typelessMap(const baseFunction &fn) const {
      occa::memory output = device_.template malloc<T2>(length());

      typelessMapTo(output, fn);

      return output;
    }

    void typelessMapTo(occa::memory output,
                       const baseFunction &fn) const {
      occa::scope arrayScope = getMapArrayScope(fn);
      arrayScope.add("occa_array_output", output);

      OCCA_JIT(arrayScope, (
        OCCA_ARRAY_TILE_FOR_LOOP {
          OCCA_ARRAY_TILE_PARALLEL_FOR_LOOP {
            occa_array_output[i] = OCCA_ARRAY_FUNCTION_CALL(i);
          }
        }
      ));
    }

    template <class T2>
    T2 typelessReduce(reductionType type,
                       const T2 &localInit,
                       const bool useLocalInit,
                       const baseFunction &fn) const {
      if (usingNativeCpuMode()) {
        return typelessCpuReduce<T2>(type, localInit, useLocalInit, fn);
      } else {
        return typelessGpuReduce<T2>(type, localInit, useLocalInit, fn);
      }
    }

    template <class T2>
    T2 typelessCpuReduce(reductionType type,
                          const T2 &localInit,
                          const bool useLocalInit,
                          const baseFunction &fn) const {
      occa::scope scope = getCpuReduceArrayScope<T2>(type, localInit, useLocalInit, fn);

      OCCA_JIT(scope, (
        for (int ompIndex = 0; ompIndex < OCCA_ARRAY_OMP_LOOP_SIZE; ++ompIndex; @outer) {
          for (int dummyIndex = 0; dummyIndex < 1; ++dummyIndex; @inner) {
            const int blockSize = (
              (occa_array_length + OCCA_ARRAY_OMP_LOOP_SIZE - 1) / OCCA_ARRAY_OMP_LOOP_SIZE
            );

            const int startIndex = ompIndex * blockSize;
            const int unsafeEndIndex = startIndex + blockSize;
            const int endIndex = occa_array_length < unsafeEndIndex ? occa_array_length : unsafeEndIndex;

            T2 localAcc = OCCA_ARRAY_REDUCTION_INIT_VALUE;

            for (int i = startIndex; i < endIndex; ++i) {
              localAcc = OCCA_ARRAY_FUNCTION_CALL(localAcc, i);
            }

            occa_array_return[ompIndex] = localAcc;
          }
        }
      ));

      return finishReturnMemoryReduction<T2>(type);
    }

    template <class T2>
    T2 typelessGpuReduce(reductionType type,
                          const T2 &localInit,
                          const bool useLocalInit,
                          const baseFunction &fn) const {
      occa::scope scope = getGpuReduceArrayScope<T2>(type, localInit, useLocalInit, fn);

      OCCA_JIT(scope, (
        for (int tileIndex = 0;
             tileIndex < occa_array_length;
             tileIndex += (OCCA_ARRAY_TILE_SIZE * OCCA_ARRAY_TILE_ITERATIONS);
             @outer) {

          @shared volatile T2 tileAcc[OCCA_ARRAY_TILE_SIZE];

          for (int localIndex = 0; localIndex < OCCA_ARRAY_TILE_SIZE; ++localIndex; @inner) {
            T2 localAcc = OCCA_ARRAY_REDUCTION_INIT_VALUE;

            for (int i = 0; i < OCCA_ARRAY_TILE_ITERATIONS; ++i) {
              const int index = tileIndex + (i * OCCA_ARRAY_TILE_ITERATIONS) + localIndex;
              if (index < occa_array_length) {
                localAcc = OCCA_ARRAY_FUNCTION_CALL(localAcc, index);
              }
            }

            tileAcc[localIndex] = localAcc;
          }

        @directive("#if OCCA_ARRAY_TILE_SIZE > 512")
          OCCA_ARRAY_SHARED_REDUCTION(512)
        @directive("#endif")

        @directive("#if OCCA_ARRAY_TILE_SIZE > 256")
          OCCA_ARRAY_SHARED_REDUCTION(256)
        @directive("#endif")

        @directive("#if OCCA_ARRAY_TILE_SIZE > 128")
          OCCA_ARRAY_SHARED_REDUCTION(128)
        @directive("#endif")

        @directive("#if OCCA_ARRAY_TILE_SIZE > 64")
          OCCA_ARRAY_SHARED_REDUCTION(64)
        @directive("#endif")

        @directive("#if OCCA_ARRAY_TILE_SIZE > 32")
          OCCA_ARRAY_SHARED_REDUCTION(32)
        @directive("#endif")

        @directive("#if OCCA_ARRAY_TILE_SIZE > 16")
          OCCA_ARRAY_SHARED_REDUCTION(16)
        @directive("#endif")

        @directive("#if OCCA_ARRAY_TILE_SIZE > 8")
          OCCA_ARRAY_SHARED_REDUCTION(8)
        @directive("#endif")

        @directive("#if OCCA_ARRAY_TILE_SIZE > 4")
          OCCA_ARRAY_SHARED_REDUCTION(4)
        @directive("#endif")

        @directive("#if OCCA_ARRAY_TILE_SIZE > 2")
          OCCA_ARRAY_SHARED_REDUCTION(2)
        @directive("#endif")

          for (int i = 0; i < OCCA_ARRAY_TILE_SIZE; ++i; @inner) {
            if (i == 0) {
              const T2 leftValue = tileAcc[0];
              const T2 rightValue = tileAcc[1];
              occa_array_return[tileIndex] = (
                OCCA_ARRAY_LOCAL_REDUCTION(leftValue, rightValue)
              );
            }
          }
        }
      ));

      return finishReturnMemoryReduction<T2>(type);
    }

    template <class T2>
    T2 finishReturnMemoryReduction(reductionType type) const {
      return functional::hostReduction<T2>(type, returnMemory);
    }
    //==================================
  };
}

#endif
