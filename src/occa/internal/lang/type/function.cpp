#include <occa/internal/lang/type/function.hpp>
#include <occa/internal/lang/variable.hpp>

namespace occa {
  namespace lang {
    function_t::function_t() :
      type_t(),
      returnType() {}

    function_t::function_t(const vartype_t &returnType_,
                           identifierToken &nameToken) :
      type_t(nameToken),
      returnType(returnType_) {}

    function_t::function_t(const vartype_t &returnType_,
                           const std::string &name_) :
      type_t(name_),
      returnType(returnType_) {}

    function_t::function_t(const function_t &other) :
      type_t(other),
      returnType(other.returnType) {

      const int count = (int) other.args.size();
      for (int i = 0; i < count; ++i) {
        args.push_back(
          &(other.args[i]->clone())
        );
      }
    }

    void function_t::free() {
      const int count = (int) args.size();
      for (int i = 0; i < count; ++i) {
        delete args[i];
      }
      args.clear();
    }

    int function_t::type() const {
      return typeType::function;
    }

    type_t& function_t::clone() const {
      return *(new function_t(*this));
    }

    dtype_t function_t::dtype() const {
      return dtype::byte;
    }

    void function_t::addArgument(const variable_t &arg) {
      args.push_back(&(arg.clone()));
    }

    void function_t::addArguments(const variableVector &args_) {
      const int count = (int) args_.size();
      for (int i = 0; i < count; ++i) {
        args.push_back(&(args_[i].clone()));
      }
    }

    void function_t::addArgumentFirst(const variable_t &arg) {
      args.insert(args.begin(), &(arg.clone()));
    }

    variable_t* function_t::removeArgument(const int index) {
      const int argCount = (int) args.size();
      if (index < 0 || argCount <= index ) {
        return NULL;
      }
      variable_t *arg = args[index];
      args.erase(args.begin() + index);
      return arg;
    }

    bool function_t::equals(const type_t &other) const {
      const function_t &other_ = other.to<function_t>();

      const int argSize = (int) args.size();
      if (argSize != (int) other_.args.size()) {
        return false;
      }
      if (returnType != other_.returnType) {
        return false;
      }

      for (int i = 0; i < argSize; ++i) {
        if (args[i]->vartype != other_.args[i]->vartype) {
          return false;
        }
      }
      return true;
    }

    void function_t::debugPrint() const {
      printer pout(io::stderr);
      printDeclaration(pout);
    }

    void function_t::printDeclaration(printer &pout) const {
      returnType.printDeclaration(pout, name());

      pout << '(';
      const std::string argIndent = pout.indentFromNewline();
      const int argCount = (int) args.size();
      for (int i = 0; i < argCount; ++i) {
        if (i) {
          pout << ",\n" << argIndent;
        }
        args[i]->printDeclaration(pout);
      }
      pout << ')';
    }
  }
}
