#include <occa/internal/lang/statement.hpp>
#include <occa/internal/lang/builtins/attributes/restrict.hpp>
#include <occa/internal/lang/qualifier.hpp>
#include <occa/internal/lang/variable.hpp>

namespace occa {
  namespace lang {
    class qualifier_t;

    namespace attributes {
      occaRestrict::occaRestrict() {}

      const std::string& occaRestrict::name() const {
        static std::string name_ = "restrict";
        return name_;
      }

      bool occaRestrict::forVariable() const {
        return true;
      }

      bool occaRestrict::forStatementType(const int sType) const {
        return (sType & statementType::declaration);
      }

      bool occaRestrict::isValid(const attributeToken_t &attr) const {
        if (attr.kwargs.size() ||
            attr.args.size()) {
          attr.printError("[@restrict] does not take arguments");
          return false;
        }
        return true;
      }

      bool occaRestrict::applyCodeTransformations(blockStatement &root,
                                                  const qualifier_t &restrictQualifier) {
        bool success = true;

        statementArray::from(root)
            .flatFilterByStatementType(
              statementType::functionDecl
              | statementType::function
            )
            .forEach([&](statement_t *smnt) {
                function_t &func = (
                  (smnt->type() & statementType::function)
                  ? smnt->to<functionStatement>().function()
                  : smnt->to<functionDeclStatement>().function()
                );

                for (auto &arg : func.args) {
                  if (!arg || !arg->hasAttribute("restrict")) {
                    continue;
                  }

                  if (!arg->vartype.isPointerType()) {
                    arg->attributes["restrict"].printError(
                      "[@restrict] can only be applied to pointer function arguments"
                    );
                    success &= false;
                    return;
                  }

                  const int pointerCount = (int) arg->vartype.pointers.size();
                  if (pointerCount) {
                    arg->vartype.pointers[pointerCount - 1] += restrictQualifier;
                  } else {
                    // Case where the type is a typedef'd pointer type
                    arg->vartype += restrictQualifier;
                  }
                }
              });

        return success;
      }
    }
  }
}
