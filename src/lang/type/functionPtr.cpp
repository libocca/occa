#include <occa/internal/lang/type/functionPtr.hpp>
#include <occa/internal/lang/variable.hpp>

namespace occa {
  namespace lang {
    functionPtr_t::functionPtr_t() :
      type_t(),
      returnType(),
      isBlock(false) {}

    functionPtr_t::functionPtr_t(const vartype_t &returnType_,
                                 identifierToken &nameToken) :
      type_t(nameToken),
      returnType(returnType_),
      isBlock(false) {}

    functionPtr_t::functionPtr_t(const vartype_t &returnType_,
                                 const std::string &name_) :
      type_t(name_),
      returnType(returnType_),
      isBlock(false) {}

    functionPtr_t::functionPtr_t(const functionPtr_t &other) :
      type_t(other),
      returnType(other.returnType),
      args(other.args),
      isBlock(other.isBlock) {}

    int functionPtr_t::type() const {
      return typeType::functionPtr;
    }

    type_t& functionPtr_t::clone() const {
      return *(new functionPtr_t(*this));
    }

    bool functionPtr_t::isPointerType() const {
      return true;
    }

    void functionPtr_t::addArgument(const variable_t &arg) {
      args.push_back(arg);
    }

    void functionPtr_t::addArguments(const variableVector &args_) {
      const int count = (int) args_.size();
      for (int i = 0; i < count; ++i) {
        args.push_back(args_[i]);
      }
    }

    dtype_t functionPtr_t::dtype() const {
      return dtype::byte;
    }

    bool functionPtr_t::equals(const type_t &other) const {
      const functionPtr_t &other_ = other.to<functionPtr_t>();

      const int argSize = (int) args.size();
      if ((isBlock != other_.isBlock)   ||
          (argSize != (int) other_.args.size())) {
        return false;
      }
      if (returnType != other_.returnType) {
        return false;
      }

      for (int i = 0; i < argSize; ++i) {
        if (args[i].vartype != other_.args[i].vartype) {
          return false;
        }
      }
      return true;
    }

    void functionPtr_t::printDeclaration(printer &pout) const {
      if (!isBlock) {
        returnType.printDeclaration(pout, "(*" + name());
      } else {
        returnType.printDeclaration(pout, "(^" + name());
      }
      pout << ')';

      pout << '(';
      const std::string argIndent = pout.indentFromNewline();
      const int argCount = (int) args.size();
      for (int i = 0; i < argCount; ++i) {
        if (i) {
          pout << ",\n" << argIndent;
        }
        args[i].printDeclaration(pout);
      }
      pout << ')';
    }
  }
}
