#ifndef OCCA_FUNCTIONAL_ARRAY_HEADER
#define OCCA_FUNCTIONAL_ARRAY_HEADER

#include <vector>

#include <occa/defines/okl.hpp>
#include <occa/dtype.hpp>
#include <occa/functional/function.hpp>
#include <occa/experimental/kernelBuilder.hpp>

namespace occa {
  class kernelArg;

  enum class reductionType {
    sum,
    multiply,
    bitOr,
    bitAnd,
    bitXor,
    boolOr,
    boolAnd,
    min,
    max
  };

  template <class TM>
  TM hostReduction(reductionType type, occa::memory mem) {
    const int entryCount = (int) mem.length();
    TM *values = new TM[entryCount];
    mem.copyTo(values);

    TM reductionValue = values[0];
    switch (type) {
      case reductionType::sum:
        for (int i = 1; i < entryCount; ++i) {
          reductionValue += values[i];
        }
        break;
      case reductionType::multiply:
        for (int i = 1; i < entryCount; ++i) {
          reductionValue *= values[i];
        }
        break;
      case reductionType::bitOr:
        for (int i = 1; i < entryCount; ++i) {
          reductionValue |= values[i];
        }
        break;
      case reductionType::bitAnd:
        for (int i = 1; i < entryCount; ++i) {
          reductionValue &= values[i];
        }
        break;
      case reductionType::bitXor:
        for (int i = 1; i < entryCount; ++i) {
          reductionValue ^= values[i];
        }
        break;
      case reductionType::boolOr:
        for (int i = 1; i < entryCount; ++i) {
          reductionValue = reductionValue || values[i];
        }
        break;
      case reductionType::boolAnd:
        for (int i = 1; i < entryCount; ++i) {
          reductionValue = reductionValue && values[i];
        }
        break;
      case reductionType::min:
        for (int i = 1; i < entryCount; ++i) {
          reductionValue = reductionValue < values[i] ? reductionValue : values[i];
        }
        break;
      case reductionType::max:
        for (int i = 1; i < entryCount; ++i) {
          reductionValue = reductionValue > values[i] ? reductionValue : values[i];
        }
        break;
      default:
        break;
    }

    delete [] values;

    return reductionValue;
  }

  template <>
  bool hostReduction<bool>(reductionType type, occa::memory mem);

  template <>
  float hostReduction<float>(reductionType type, occa::memory mem);

  template <>
  double hostReduction<double>(reductionType type, occa::memory mem);

  template <class TM>
  class array {
    template <class TM2>
    friend class array;

  private:
    occa::memory memory_;
    int tileSize;
    int tileIterations;

    // Buffer memory
    mutable occa::memory returnMemory;

    template <class ReturnType>
    void setupReturnMemory(const ReturnType &value) const {
      size_t bytes = sizeof(ReturnType);
      if (bytes > returnMemory.size()) {
        returnMemory = getDevice().malloc(bytes);
      }
      returnMemory.setDtype(dtype::get<ReturnType>());
      returnMemory.copyFrom(&value, bytes);
    }

    template <class ReturnType>
    void setupReturnMemoryArray(const int size) const {
      size_t bytes = sizeof(ReturnType) * size;
      if (bytes > returnMemory.size()) {
        returnMemory = getDevice().malloc(bytes);
      }
      returnMemory.setDtype(dtype::get<ReturnType>());
    }

    template <class ReturnType>
    void setReturnValue(ReturnType &value) const {
      size_t bytes = sizeof(ReturnType);
      returnMemory.copyTo(&value, bytes);
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

      return occa::scope({
        {"occa_array_ptr", memory_},
        {"occa_array_length", arrayLength},
        {"occa_array_return", returnMemory}
      }, {
        {"defines/TM", dtype::get<TM>().name()},
        {"defines/OCCA_ARRAY_TILE_SIZE", safeTileSize},
        {"defines/OCCA_ARRAY_TILE_ITERATIONS", safeTileIterations},
        {"defines/OCCA_ARRAY_FUNCTION(VALUE, INDEX, VALUES_PTR)", buildMapFunctionCall(fn)},
        {"defines/OCCA_ARRAY_TILE_FOR_LOOP", tileForLoop},
        {"defines/OCCA_ARRAY_TILE_PARALLEL_FOR_LOOP", parallelForLoop},
        {"functions/occa_array_function", fn}
      }) + fn.scope;
    }

    template <class TM2>
    occa::scope getCpuReduceArrayScope(reductionType type,
                                       const TM2 &localInit,
                                       const bool useLocalInit,
                                       const baseFunction &fn) const {
      const int arrayLength = (int) length();
      const int ompLoopSize = 128;

      setupReturnMemoryArray<TM2>(ompLoopSize);

      occa::json props({
        {"defines/TM", dtype::get<TM>().name()},
        {"defines/TM2", dtype::get<TM2>().name()},
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

      return occa::scope({
        {"occa_array_ptr", memory_},
        {"occa_array_length", arrayLength},
        {"occa_array_return", returnMemory}
      }, props) + fn.scope;
    }

    template <class TM2>
    occa::scope getGpuReduceArrayScope(reductionType type,
                                       const TM2 &localInit,
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

      setupReturnMemoryArray<TM2>(localReductionCount);

      occa::json props({
        {"defines/TM", dtype::get<TM>().name()},
        {"defines/TM2", dtype::get<TM2>().name()},
        {"defines/OCCA_ARRAY_TILE_SIZE", safeTileSize},
        {"defines/OCCA_ARRAY_TILE_ITERATIONS", safeTileIterations},
        {"defines/OCCA_ARRAY_FUNCTION(ACC, VALUE, INDEX, VALUES_PTR)", buildReduceFunctionCall(fn)},
        {"defines/OCCA_ARRAY_LOCAL_REDUCTION(LEFT_VALUE, RIGHT_VALUE)", buildLocalReductionOperation(type)},
        {"defines/OCCA_ARRAY_SHARED_REDUCTION(BOUNDS)",
         "for (int i = 0; i < OCCA_ARRAY_TILE_SIZE; ++i; @inner) {"
         "  if (i < BOUNDS) {"
         "    const TM2 leftValue = tileAcc[i];"
         "    const TM2 rightValue = tileAcc[i + BOUNDS];"
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

      return occa::scope({
        {"occa_array_ptr", memory_},
        {"occa_array_length", arrayLength},
        {"occa_array_return", returnMemory}
      }, props) + fn.scope;
    }

    std::string buildMapFunctionCall(const baseFunction &fn) const {
      return buildFunctionCall(fn, true);
    }

    std::string buildReduceFunctionCall(const baseFunction &fn) const {
      return buildFunctionCall(fn, false);
    }

    std::string buildFunctionCall(const baseFunction &fn,
                                  const bool forMapFunction) const {
      std::string call = "occa_array_function(";

      // Add the required arguments
      int requiredArgumentCount;
      if (forMapFunction) {
        requiredArgumentCount = 1;
        call += "VALUE";
      } else {
        requiredArgumentCount = 2;
        call += "ACC, VALUE";
      }

      // Add the optional arguments
      const std::string optionalArgumentNames[2] = {
        "INDEX", "VALUES_PTR"
      };
      for (int i = 0; i < (fn.argumentCount() - requiredArgumentCount); ++i) {
        call += ", ";
        call += optionalArgumentNames[i];
      }

      // Add the scope-injected arguments
      for (const scopeKernelArg &arg : fn.scope.args) {
        call += ", ";
        call += arg.name;
      }

      call += ")";

      return call;
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
        "for (int tileIndex = 0; tileIndex < occa_array_length; tileIndex += OCCA_ARRAY_TILE_ITERATIONS;"
        " @tile(OCCA_ARRAY_TILE_SIZE * OCCA_ARRAY_TILE_ITERATIONS, @outer, @inner, check=false))"
      );

      parallelForLoop = (
        "for (int i = tileIndex; i < tileIndex + OCCA_ARRAY_TILE_ITERATIONS; ++i)"
        "  if (i < occa_array_length)"
      );
    }

    void buildGpuMapTiledForLoops(std::string &tileForLoop,
                                  std::string &parallelForLoop) const {
      tileForLoop = (
        "for (int tileIndex = 0; tileIndex < occa_array_length; tileIndex += OCCA_ARRAY_TILE_SIZE;"
        " @tile(OCCA_ARRAY_TILE_SIZE * OCCA_ARRAY_TILE_ITERATIONS, @outer, @inner, check=false))"
      );

      parallelForLoop = (
        "for (int i = tileIndex; i < tileIndex + (OCCA_ARRAY_TILE_SIZE * OCCA_ARRAY_TILE_ITERATIONS); i += OCCA_ARRAY_TILE_ITERATIONS)"
        "  if (i < occa_array_length)"
      );
    }

    std::string buildReductionInitValue(reductionType type) const {
      switch (type) {
        case reductionType::sum:
          return "0";
        case reductionType::multiply:
          return "1";
        case reductionType::bitOr:
          return "0";
        case reductionType::bitAnd:
          return "occa_array_ptr[0]";
        case reductionType::bitXor:
          return "0";
        case reductionType::boolOr:
          return "0";
        case reductionType::boolAnd:
          return "occa_array_ptr[0]";
        case reductionType::min:
          return "occa_array_ptr[0]";
        case reductionType::max:
          return "occa_array_ptr[0]";
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

  public:
    array() :
      tileSize(-1),
      tileIterations(-1) {}

    array(const dim_t size) :
      tileSize(-1),
      tileIterations(-1) {

      memory_ = occa::malloc<TM>(size);
    }

    array(occa::device device, const dim_t size) :
      tileSize(-1),
      tileIterations(-1) {

      memory_ = device.malloc<TM>(size);
    }

    array(occa::memory mem) :
      memory_(mem),
      tileSize(-1),
      tileIterations(-1) {}

    array(const array<TM> &other) :
      memory_(other.memory_),
      tileSize(other.tileSize),
      tileIterations(other.tileIterations) {}

    array& operator = (const array<TM> &other) {
      memory_ = other.memory_;
      tileSize = other.tileSize;
      tileIterations = other.tileIterations;

      return *this;
    }

    bool usingNativeCpuMode() const {
      const std::string &mode = getDevice().mode();
      return (mode == "Serial" || mode == "OpenMP");
    }

    void setTileSize(const int tileSize_) {
      if (tileSize_ >= 0) {
        tileSize = tileSize_;
      }
    }

    void setTileIterations(const int tileIterations_) {
      if (tileIterations_ >= 0) {
        tileIterations = tileIterations_;
      }
    }

    //---[ Memory methods ]-------------
    bool isInitialized() const {
      return memory_.isInitialized();
    }

    occa::device getDevice() const {
      return memory_.getDevice();
    }

    occa::memory memory() const {
      return memory_;
    }

    operator occa::memory () const {
      return memory_;
    }

    occa::dtype_t dtype() const {
      return memory_.dtype();
    }

    operator kernelArg() const {
      return memory_;
    }

    udim_t length() const {
      return memory_.length<TM>();
    }

    array clone() const {
      return array(memory_.clone());
    }

    void copyFrom(const TM *src,
                  const dim_t entries = -1) {
      const dim_t safeEntries = (
        entries <= 0
        ? length()
        : entries
      );

      memory_.copyFrom(src, safeEntries * sizeof(TM));
    }

  void copyFrom(const occa::memory src,
                const dim_t entries = -1) {
      const dim_t safeEntries = (
        entries <= 0
        ? length()
        : entries
      );

      memory_.copyFrom(src, safeEntries * sizeof(TM));
    }

    void copyTo(TM *dest,
                const dim_t entries = -1) const {
      const dim_t safeEntries = (
        entries <= 0
        ? length()
        : entries
      );

      memory_.copyTo(dest, safeEntries * sizeof(TM));
    }

    void copyTo(occa::memory dest,
                const dim_t entries = -1) const {
      const dim_t safeEntries = (
        entries <= 0
        ? length()
        : entries
      );

      memory_.copyTo(dest, safeEntries * sizeof(TM));
    }
    //==================================

  private:
    //---[ Lambda methods ]-------------
    bool typelessEvery(const baseFunction &fn) const {
      bool returnValue = true;

      setupReturnMemory(returnValue);

      OCCA_JIT(getMapArrayScope(fn), (
        OCCA_ARRAY_TILE_FOR_LOOP {
          OCCA_ARRAY_TILE_PARALLEL_FOR_LOOP {
            if (!OCCA_ARRAY_FUNCTION(occa_array_ptr[i], i, occa_array_ptr)) {
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

    TM typelessFind(const TM &default_,
                    const baseFunction &fn) const {
      TM returnValue = default_;

      setupReturnMemory(returnValue);

      OCCA_JIT(getMapArrayScope(fn), (
        OCCA_ARRAY_TILE_FOR_LOOP {
          OCCA_ARRAY_TILE_PARALLEL_FOR_LOOP {
            const TM &value = occa_array_ptr[i];
            if (OCCA_ARRAY_FUNCTION(value, i, occa_array_ptr)) {
              occa_array_return[0] = value;
            }
          }
        }
      ));

      setReturnValue(returnValue);

      return returnValue;
    }

    int typelessFindIndex(const baseFunction &fn) const {
      int returnValue = -1;

      setupReturnMemory(returnValue);

      OCCA_JIT(getMapArrayScope(fn), (
        OCCA_ARRAY_TILE_FOR_LOOP {
          OCCA_ARRAY_TILE_PARALLEL_FOR_LOOP {
            if (OCCA_ARRAY_FUNCTION(occa_array_ptr[i], i, occa_array_ptr)) {
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
            OCCA_ARRAY_FUNCTION(occa_array_ptr[i], i, occa_array_ptr);
          }
        }
      ));
    }

    template <class TM2>
    array<TM2> typelessMap(const baseFunction &fn) const {
      array<TM2> output = getDevice().template malloc<TM2>(length());
      return typelessMapTo(output, fn);
    }

    template <class TM2>
    array<TM2> typelessMapTo(occa::array<TM2> &output,
                             const baseFunction &fn) const {
      occa::scope arrayScope = getMapArrayScope(fn);
      arrayScope.add("occa_array_output", output.memory_);

      OCCA_JIT(arrayScope, (
        OCCA_ARRAY_TILE_FOR_LOOP {
          OCCA_ARRAY_TILE_PARALLEL_FOR_LOOP {
            occa_array_output[i] = OCCA_ARRAY_FUNCTION(occa_array_ptr[i], i, occa_array_ptr);
          }
        }
      ));

      return output;
    }

    template <class TM2>
    TM2 typelessReduce(reductionType type,
                       const TM2 &localInit,
                       const bool useLocalInit,
                       const baseFunction &fn) const {
      if (usingNativeCpuMode()) {
        return typelessCpuReduce<TM2>(type, localInit, useLocalInit, fn);
      } else {
        return typelessGpuReduce<TM2>(type, localInit, useLocalInit, fn);
      }
    }

    template <class TM2>
    TM2 typelessCpuReduce(reductionType type,
                          const TM2 &localInit,
                          const bool useLocalInit,
                          const baseFunction &fn) const {
      OCCA_JIT(getCpuReduceArrayScope<TM2>(type, localInit, useLocalInit, fn), (
        for (int ompIndex = 0; ompIndex < OCCA_ARRAY_OMP_LOOP_SIZE; ++ompIndex; @outer) {
          for (int dummyIndex = 0; dummyIndex < 1; ++dummyIndex; @inner) {
            const int blockSize = (
              (occa_array_length + OCCA_ARRAY_OMP_LOOP_SIZE - 1) / OCCA_ARRAY_OMP_LOOP_SIZE
            );

            const int startIndex = ompIndex * blockSize;
            const int unsafeEndIndex = startIndex + blockSize;
            const int endIndex = occa_array_length < unsafeEndIndex ? occa_array_length : unsafeEndIndex;

            TM2 localAcc = OCCA_ARRAY_REDUCTION_INIT_VALUE;

            for (int i = startIndex; i < endIndex; ++i) {
              localAcc = OCCA_ARRAY_FUNCTION(localAcc, occa_array_ptr[i], i, occa_array_ptr);
            }

            occa_array_return[ompIndex] = localAcc;
          }
        }
      ));

      return finishReturnMemoryReduction<TM2>(type);
    }

    template <class TM2>
    TM2 typelessGpuReduce(reductionType type,
                          const TM2 &localInit,
                          const bool useLocalInit,
                          const baseFunction &fn) const {
      OCCA_JIT(getGpuReduceArrayScope<TM2>(type, localInit, useLocalInit, fn), (
        for (int tileIndex = 0;
             tileIndex < occa_array_length;
             tileIndex += (OCCA_ARRAY_TILE_SIZE * OCCA_ARRAY_TILE_ITERATIONS);
             @outer) {

          @shared volatile TM2 tileAcc[OCCA_ARRAY_TILE_SIZE];

          for (int localIndex = 0; localIndex < OCCA_ARRAY_TILE_SIZE; ++localIndex; @inner) {
            TM2 localAcc = OCCA_ARRAY_REDUCTION_INIT_VALUE;

            for (int i = 0; i < OCCA_ARRAY_TILE_ITERATIONS; ++i) {
              const int index = tileIndex + (i * OCCA_ARRAY_TILE_ITERATIONS) + localIndex;
              if (index < occa_array_length) {
                localAcc = OCCA_ARRAY_FUNCTION(localAcc, occa_array_ptr[index], index, occa_array_ptr);
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
              const TM2 leftValue = tileAcc[0];
              const TM2 rightValue = tileAcc[1];
              occa_array_return[tileIndex] = (
                OCCA_ARRAY_LOCAL_REDUCTION(leftValue, rightValue)
              );
            }
          }
        }
      ));

      return finishReturnMemoryReduction<TM2>(type);
    }

    template <class TM2>
    TM2 finishReturnMemoryReduction(reductionType type) const {
      return hostReduction<TM2>(type, returnMemory);
    }

  public:
    bool every(const occa::function<bool(TM)> &fn) const {
      return typelessEvery(fn);
    }

    bool every(const occa::function<bool(TM, int)> &fn) const {
      return typelessEvery(fn);
    }

    bool every(const occa::function<bool(TM, int, const TM*)> &fn) const {
      return typelessEvery(fn);
    }

    bool some(const occa::function<bool(TM)> &fn) const {
      return typelessSome(fn);
    }

    bool some(const occa::function<bool(TM, int)> &fn) const {
      return typelessSome(fn);
    }

    bool some(const occa::function<bool(TM, int, const TM*)> &fn) const {
      return typelessSome(fn);
    }

    TM find(const TM &default_,
            const occa::function<bool(TM)> &fn) const {
      return typelessFind(default_, fn);
    }

    TM find(const TM &default_,
            const occa::function<bool(TM, int)> &fn) const {
      return typelessFind(default_, fn);
    }

    TM find(const TM &default_,
            const occa::function<bool(TM, int, const TM*)> &fn) const {
      return typelessFind(default_, fn);
    }

    int findIndex(const occa::function<bool(TM)> &fn) const {
      return typelessFindIndex(fn);
    }

    int findIndex(const occa::function<bool(TM, int)> &fn) const {
      return typelessFindIndex(fn);
    }

    int findIndex(const occa::function<bool(TM, int, const TM*)> &fn) const {
      return typelessFindIndex(fn);
    }

    void forEach(const occa::function<void(TM)> &fn) const {
      return typelessForEach(fn);
    }

    void forEach(const occa::function<void(TM, int)> &fn) const {
      return typelessForEach(fn);
    }

    void forEach(const occa::function<void(TM, int, const TM*)> &fn) const {
      return typelessForEach(fn);
    }

    template <class TM2>
    array<TM2> map(const occa::function<TM2(TM)> &fn) const {
      return typelessMap<TM2>(fn);
    }

    template <class TM2>
    array<TM2> map(const occa::function<TM2(TM, int)> &fn) const {
      return typelessMap<TM2>(fn);
    }

    template <class TM2>
    array<TM2> map(const occa::function<TM2(TM, int, const TM*)> &fn) const {
      return typelessMap<TM2>(fn);
    }

    template <class TM2>
    array<TM2> mapTo(occa::array<TM2> &output,
                     const occa::function<TM2(TM)> &fn) const {
      return typelessMapTo<TM2>(output, fn);
    }

    template <class TM2>
    array<TM2> mapTo(occa::array<TM2> &output,
                     const occa::function<TM2(TM, int)> &fn) const {
      return typelessMapTo<TM2>(output, fn);
    }

    template <class TM2>
    array<TM2> mapTo(occa::array<TM2> &output,
                     const occa::function<TM2(TM, int, const TM*)> &fn) const {
      return typelessMapTo<TM2>(output, fn);
    }

    template <class TM2>
    TM2 reduce(reductionType type,
               const occa::function<TM2(TM2, TM)> &fn) const {
      return typelessReduce<TM2>(type, TM2(), false, fn);
    }

    template <class TM2>
    TM2 reduce(reductionType type,
               const occa::function<TM2(TM2, TM, int)> &fn) const {
      return typelessReduce<TM2>(type, TM2(), false, fn);
    }

    template <class TM2>
    TM2 reduce(reductionType type,
               occa::function<TM2(TM2, TM, int, const TM*)> fn) const {
      return typelessReduce<TM2>(type, TM2(), false, fn);
    }

    template <class TM2>
    TM2 reduce(reductionType type,
               const TM2 &localInit,
               const occa::function<TM2(TM2, TM)> &fn) const {
      return typelessReduce<TM2>(type, localInit, true, fn);
    }

    template <class TM2>
    TM2 reduce(reductionType type,
               const TM2 &localInit,
               const occa::function<TM2(TM2, TM, int)> &fn) const {
      return typelessReduce<TM2>(type, localInit, true, fn);
    }

    template <class TM2>
    TM2 reduce(reductionType type,
               const TM2 &localInit,
               occa::function<TM2(TM2, TM, int, const TM*)> fn) const {
      return typelessReduce<TM2>(type, localInit, true, fn);
    }
    //==================================

    //---[ Utility methods ]------------
    TM operator [] (const dim_t index) const {
      TM value;
      memory_.copyTo(&value,
                     1 * sizeof(TM),
                     index * sizeof(TM));
      return value;
    }

    array slice(const dim_t offset,
                const dim_t count = -1) const {
      return array(
        memory_.slice(offset, count)
      );
    }

    array concat(const array &other) const {
      const udim_t bytes1 = memory_.size();
      const udim_t bytes2 = other.memory_.size();

      occa::memory ret = getDevice().template malloc<TM>(length() + other.length());
      ret.copyFrom(memory_, bytes1, 0);
      ret.copyFrom(other.memory_, bytes2, bytes1);

      return array(ret);
    }

    array fill(const TM &fillValue) {
      occa::scope fnScope({
        {"fillValue", fillValue}
      });

      return mapTo<TM>(
        *this,
        OCCA_FUNCTION(fnScope, [=](TM value) -> TM {
          return fillValue;
        })
      );
    }

    bool includes(const TM &target) const {
      occa::scope fnScope({
        {"target", target}
      });

      return some(
        OCCA_FUNCTION(fnScope, [=](TM value) -> bool {
          return target == value;
        })
      );
    }

    dim_t indexOf(const TM &target) const {
      occa::scope fnScope({
        {"target", target}
      });

      const int _length = (int) length();

      const int returnValue = reduce<int>(
        reductionType::min,
        (int) _length,
        OCCA_FUNCTION(fnScope, [=](int foundIndex, TM value, int index) -> int {
          if ((target != value) || (foundIndex <= index)) {
            return foundIndex;
          }
          return index;
        })
      );

      return returnValue < _length ? returnValue : -1;
    }

    dim_t lastIndexOf(const TM &target) const {
      occa::scope fnScope({
        {"target", target}
      });

      return reduce<int>(
        reductionType::max,
        -1,
        OCCA_FUNCTION(fnScope, [=](int foundIndex, TM value, int index) -> int {
          if ((target != value) || (foundIndex >= index)) {
            return foundIndex;
          }
          return index;
        })
      );
    }

    template <class TM2>
    array<TM2> cast() const {
      occa::scope fnScope({}, {
        {"defines/TM2", dtype::get<TM>().name()}
      });

      return map<TM2>(
        OCCA_FUNCTION(fnScope, [=](TM value) -> TM2 {
          return (TM2) value;
        })
      );
    }

    array reverse() const {
      const int size = (int) length();

      occa::scope fnScope({
        {"size", size}
      });

      return map<TM>(
        OCCA_FUNCTION(fnScope, [=](TM value, int index, const TM *values) -> TM {
          return values[size - index - 1];
        })
      );
    }

    array shiftLeft(const int offset,
                    const TM emptyValue = TM()) const {
      if (offset == 0) {
        return clone();
      }

      const int size = (int) length();

      occa::scope fnScope({
        {"size", size},
        {"offset", offset},
        {"emptyValue", emptyValue},
      });

      return map<TM>(
        OCCA_FUNCTION(fnScope, [=](TM value, int index, const TM *values) -> TM {
          if (index < (size - offset)) {
            return values[index + offset];
          } else {
            return emptyValue;
          }
        })
      );
    }

    array shiftRight(const int offset,
                     const TM emptyValue = TM()) const {
      if (offset == 0) {
        return clone();
      }

      const int size = (int) length();

      occa::scope fnScope({
        {"size", size},
        {"offset", offset},
        {"emptyValue", emptyValue},
      });

      return map<TM>(
        OCCA_FUNCTION(fnScope, [=](TM value, int index, const TM *values) -> TM {
          if (index >= offset) {
            return values[index - offset];
          } else {
            return emptyValue;
          }
        })
      );
    }

    TM max() const {
      return reduce<TM>(
        reductionType::max,
        OCCA_FUNCTION({}, [=](TM currentMax, TM value) -> TM {
          return currentMax > value ? currentMax : value;
        })
      );
    }

    TM min() const {
      return reduce<TM>(
        reductionType::min,
        OCCA_FUNCTION({}, [=](TM currentMin, TM value) -> TM {
          return currentMin < value ? currentMin : value;
        })
      );
    }
    //==================================

    //---[ Linear Algebra Methods ]-----
    TM dotProduct(const array<TM> &other) {
      occa::scope fnScope({
        {"other", other}
      });

      return reduce<TM>(
        reductionType::sum,
        OCCA_FUNCTION(fnScope, [=](TM acc, TM value, int index) -> TM {
          return acc + (value * other[index]);
        })
      );
    }

    array clamp(const TM minValue,
                const TM maxValue) {
      occa::scope fnScope({
        {"minValue", minValue},
        {"maxValue", maxValue},
      });

      return map<TM>(
        OCCA_FUNCTION(fnScope, [=](TM value) -> TM {
          const TM valueWithMaxClamp = value > maxValue ? maxValue : value;
          return valueWithMaxClamp < minValue ? minValue : valueWithMaxClamp;
        })
      );
    }

    array clampMin(const TM minValue) {
      occa::scope fnScope({
        {"minValue", minValue},
      });

      return map<TM>(
        OCCA_FUNCTION(fnScope, [=](TM value) -> TM {
          return value < minValue ? minValue : value;
        })
      );
    }

    array clampMax(const TM maxValue) {
      occa::scope fnScope({
        {"maxValue", maxValue},
      });

      return map<TM>(
        OCCA_FUNCTION(fnScope, [=](TM value) -> TM {
          return value > maxValue ? maxValue : value;
        })
      );
    }
    //==================================
  };

  template <>
  inline bool hostReduction<bool>(reductionType type, occa::memory mem) {
    const int entryCount = (int) mem.length();
    bool *values = new bool[entryCount];
    mem.copyTo(values);

    bool reductionValue = values[0];
    switch (type) {
      case reductionType::bitOr:
        for (int i = 1; i < entryCount; ++i) {
          reductionValue |= values[i];
        }
        break;
      case reductionType::bitAnd:
        for (int i = 1; i < entryCount; ++i) {
          reductionValue &= values[i];
        }
        break;
      case reductionType::bitXor:
        for (int i = 1; i < entryCount; ++i) {
          reductionValue ^= values[i];
        }
        break;
      case reductionType::boolOr:
        for (int i = 1; i < entryCount; ++i) {
          reductionValue = reductionValue || values[i];
        }
        break;
      case reductionType::boolAnd:
        for (int i = 1; i < entryCount; ++i) {
          reductionValue = reductionValue && values[i];
        }
        break;
      case reductionType::sum:
      case reductionType::multiply:
        OCCA_FORCE_ERROR("Arithmetic operations not implemented for occa::array<bool>");
        break;
      case reductionType::min:
      case reductionType::max:
        OCCA_FORCE_ERROR("Comparison operations not implemented for occa::array<bool>");
        break;
      default:
        break;
    }

    delete [] values;

    return reductionValue;
  }

  template <>
  inline float hostReduction<float>(reductionType type, occa::memory mem) {
    const int entryCount = (int) mem.length();
    float *values = new float[entryCount];
    mem.copyTo(values);

    float reductionValue = values[0];
    switch (type) {
      case reductionType::sum:
        for (int i = 1; i < entryCount; ++i) {
          reductionValue += values[i];
        }
        break;
      case reductionType::multiply:
        for (int i = 1; i < entryCount; ++i) {
          reductionValue *= values[i];
        }
        break;
      case reductionType::min:
        for (int i = 1; i < entryCount; ++i) {
          reductionValue = reductionValue < values[i] ? reductionValue : values[i];
        }
        break;
      case reductionType::max:
        for (int i = 1; i < entryCount; ++i) {
          reductionValue = reductionValue > values[i] ? reductionValue : values[i];
        }
        break;
      case reductionType::bitOr:
      case reductionType::bitAnd:
      case reductionType::bitXor:
        OCCA_FORCE_ERROR("Bit operations not implemented for occa::array<float>");
        break;
      case reductionType::boolOr:
      case reductionType::boolAnd:
        OCCA_FORCE_ERROR("Boolean operations not implemented for occa::array<double>");
        break;
      default:
        break;
    }

    delete [] values;

    return reductionValue;
  }

  template <>
  inline double hostReduction<double>(reductionType type, occa::memory mem) {
    const int entryCount = (int) mem.length();
    double *values = new double[entryCount];
    mem.copyTo(values);

    double reductionValue = values[0];
    switch (type) {
      case reductionType::sum:
        for (int i = 1; i < entryCount; ++i) {
          reductionValue += values[i];
        }
        break;
      case reductionType::multiply:
        for (int i = 1; i < entryCount; ++i) {
          reductionValue *= values[i];
        }
        break;
      case reductionType::min:
        for (int i = 1; i < entryCount; ++i) {
          reductionValue = reductionValue < values[i] ? reductionValue : values[i];
        }
        break;
      case reductionType::max:
        for (int i = 1; i < entryCount; ++i) {
          reductionValue = reductionValue > values[i] ? reductionValue : values[i];
        }
        break;
      case reductionType::bitOr:
      case reductionType::bitAnd:
      case reductionType::bitXor:
        OCCA_FORCE_ERROR("Bit operations not implemented for occa::array<double>");
        break;
      case reductionType::boolOr:
      case reductionType::boolAnd:
        OCCA_FORCE_ERROR("Boolean operations not implemented for occa::array<double>");
        break;
      default:
        break;
    }

    delete [] values;

    return reductionValue;
  }
}

#endif
