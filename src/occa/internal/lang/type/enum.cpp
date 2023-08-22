#include <occa/internal/lang/type/enum.hpp>
#include <occa/internal/lang/enumerator.hpp>
#include <occa/internal/lang/expr.hpp>
#include <occa/dtype.hpp>

namespace occa {
  namespace lang {

    enum_t::enum_t() :
      type_t() {}

    enum_t::enum_t(identifierToken &nameToken) :
      type_t(nameToken) {}

    enum_t::enum_t(const enum_t &other) :
      type_t(other) {

        const int count = (int) other.enumerators.size();
        for (int i = 0; i < count; ++i) {
          enumerators.push_back(
            other.enumerators[i].clone()
          );
        }
      }

    int enum_t::type() const {
      return typeType::enum_;
    }

    type_t& enum_t::clone() const {
      return *(new enum_t(*this));
    }

    dtype_t enum_t::dtype() const {
      dtype_t dtype_;
      const int enumeratorsCount = (int) enumerators.size();
      for (int i =0; i < enumeratorsCount; ++i) {
        const enumerator_t &enumerator_ = enumerators[i];
        dtype_.addEnumerator(enumerator_.source->value);
      }
      return dtype_;
    }

    void enum_t::addEnumerator(enumerator_t &enumerator_) {
      enumerators.push_back(
                            enumerator_t(
                              (identifierToken*) enumerator_.source->clone(),
                              exprNode::clone(enumerator_.expr)
                              )
                            );
    }

    void enum_t::addEnumerators(enumeratorVector &enumerators_) {
      const int enumeratorCount = (int) enumerators_.size();
      for (int i = 0; i < enumeratorCount; ++i) {
        addEnumerator(enumerators_[i]);
      }
    }

    void enum_t::debugPrint() const {
      printer pout(io::stderr);
      printDeclaration(pout);
    }

    void enum_t::printDeclaration(printer &pout) const {
      const std::string name_ = name();
      if (name_.size()) {
        pout << name_ << ' ';
      }

      const int enumeratorsCount = (int) enumerators.size();
      if (!enumeratorsCount) {
        pout << "{}";
      } else {
        pout << "{\n";
        pout.addIndentation();
        pout.printIndentation();
        for (int i = 0; i < enumeratorsCount; ++i) {
          const enumerator_t &enumerator_ = enumerators[i];
            if (i) {
              pout << ", \n";
              pout.printIndentation();
            }
            pout << enumerator_.source->value;
            if (enumerator_.expr) {
              pout << "=";
              pout << enumerator_.expr;
            }
          }
        pout << "\n";
        pout.removeIndentation();
        pout.printIndentation();
        pout << "}";
      }
    }
  }
}
