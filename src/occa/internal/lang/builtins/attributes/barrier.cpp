#include <occa/internal/lang/expr.hpp>
#include <occa/internal/lang/parser.hpp>
#include <occa/internal/lang/statement.hpp>
#include <occa/internal/lang/variable.hpp>
#include <occa/internal/lang/builtins/attributes/barrier.hpp>

namespace occa {
  namespace lang {
    namespace attributes {

      barrier::barrier() {}

      const std::string& barrier::name() const {
        static std::string name_ = "barrier";
        return name_;
      }

      bool barrier::forStatementType(const int sType) const {
        return (sType & statementType::empty);
      }

      bool barrier::isValid(const attributeToken_t &attr) const {
        return getBarrierSyncType(&attr) != invalid;
      }

      barrier::SyncType barrier::getBarrierSyncType(const attributeToken_t *attr) {
        if (!attr) {
          return invalid;
        }

        if (attr->kwargs.size()) {
          attr->printError("[@barrier] does not take kwargs");
          return invalid;
        }

        const int argCount = (int) attr->args.size();

        // Default to syncDefault
        if (!argCount) {
          return syncDefault;
        }

        if (argCount > 1) {
          attr->printError("[@barrier] takes at most one argument");
          return invalid;
        }
        if ((argCount == 1) &&
            (!attr->args[0].expr ||
             attr->args[0].expr->type() != exprNodeType::string)) {
          attr->printError("[@barrier] must have no arguments"
                          " or have one string argument");
          return invalid;
        }

        const std::string barrierType = (
          attr->args[0].expr->to<stringNode>().value
        );

        if (barrierType == "warp") {
          return syncWarp;
        }

        attr->printError(
          "[@barrier] has an invalid barrier type: " + barrierType
        );
        return invalid;
      }
    }
  }
}
