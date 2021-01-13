#include <algorithm>
#include <sstream>

#include <occa/types.hpp>
#include <occa/internal/utils/lex.hpp>
#include <occa/internal/utils/string.hpp>
#include <occa/experimental/array/linalg.hpp>

namespace occa {
  namespace linalg {
    // "v0[i] = c1 * (v0[i] + v1[i]);"
    kernelBuilder customLinearMethod(const std::string &kernelName,
                                     const std::string &formula,
                                     const occa::json &props) {

      kernelBuilder builder;

      // Extract input and constant counts
      // If formula is correct
      //   - minInput should go down to 0
      //   - maxOutput should go up to at least 0
      intVector inputs, constants;
      int minInput = 1, maxInput = -1;
      int minConstant = 0, maxConstant = -1;

      const char *c = formula.c_str();
      while (*c) {
        // Input vectors/constants are of the format [cv][0-9]+
        if ((*c == 'v') || (*c == 'c')) {
          intVector &vec = (*c == 'v') ? inputs : constants;
          int &minVal = (*c == 'v') ? minInput : minConstant;
          int &maxVal = (*c == 'v') ? maxInput : maxConstant;

          const char *cStart = ++c;
          lex::skipFrom(c, lex::numberCharset);
          // This is not the 'v' or 'c' you're looking for
          if (cStart == c) {
            ++c;
            continue;
          }

          const int idx = occa::parseInt(std::string(cStart, c - cStart));
          if (std::find(vec.begin(), vec.end(), idx) == vec.end()) {
            vec.push_back(idx);
            if (minVal > idx) {
              minVal = idx;
            }
            if (maxVal < idx) {
              maxVal = idx;
            }
          }
        }
        ++c;
      }

      const int inputCount = (int) inputs.size();
      const int constantCount = (int) constants.size();

      // Make sure vectors and constants are propertly labeled:
      //   v0, v1, v2, ...
      //   c0, c1, c2, ...
      OCCA_ERROR("Minimum vector index must be 0 (the output)",
                 minInput == 0);
      OCCA_ERROR("Cannot skip vector indices, found index "
                 << maxInput << " but only " << inputCount << " vector inputs",
                 maxInput == (inputCount - 1));

      OCCA_ERROR("Minimum constant index must be 0",
                 minConstant == 0);
      OCCA_ERROR("Cannot skip constant indices, found index "
                 << maxConstant << " but only " << constantCount << " constants inputs",
                 maxConstant == (constantCount - 1));

      // Make sure we have all the defines needed
      OCCA_ERROR("TILESIZE must be defined",
                 props.has("defines/TILESIZE"));
      for (int i = 0; i < inputCount; ++i) {
        const std::string vtype = "VTYPE" + toString(i);
        OCCA_ERROR("Type " << vtype << " must be defined",
                   props.has("defines/" + vtype));
      }
      for (int i = 0; i < constantCount; ++i) {
        const std::string ctype = "CTYPE" + toString(i);
        OCCA_ERROR("Type " << ctype << " must be defined",
                   props.has("defines/" + ctype));
      }

      std::stringstream ss;

      // Setup arguments
      ss << "@kernel void " << kernelName << "(const int entries,\n";
      for (int i = 0; i < constantCount; ++i) {
        ss << "                  const CTYPE" << i << " c" << i;
        if ((i < (constantCount - 1)) || inputCount) {
          ss << ",\n";
        }
      }
      for (int i = 0; i < inputCount; ++i) {
        ss << "                 "
           << (i != 0 ? "const " : "      ")
           << "VTYPE" << i << " *v" << i;
        if (i < (inputCount - 1)) {
          ss << ",\n";
        }
      }
      // Setup body
      ss << ") {\n"
        "  for (int i = 0; i < entries; ++i; @tile(TILESIZE, @outer, @inner)) {\n"
        "    " << formula << "\n"
        "  }\n"
        "}\n";

      return kernelBuilder::fromString(ss.str(), kernelName, props);
    }
  }
}
