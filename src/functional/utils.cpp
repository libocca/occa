#include <occa/defines.hpp>
#include <occa/functional/utils.hpp>

namespace occa {
  namespace functional {
    template <>
    bool hostReduction<bool>(reductionType type, occa::memory mem) {
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
    float hostReduction<float>(reductionType type, occa::memory mem) {
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
    double hostReduction<double>(reductionType type, occa::memory mem) {
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
}
