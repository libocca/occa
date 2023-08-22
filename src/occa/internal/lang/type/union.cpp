#include <occa/internal/lang/type/union.hpp>
#include <occa/internal/lang/variable.hpp>
#include <occa/dtype.hpp>

namespace occa {
  namespace lang {
    union_t::union_t() :
      type_t() {}

    union_t::union_t(identifierToken &nameToken) :
      type_t(nameToken) {}

    union_t::union_t(const union_t &other) :
      type_t(other) {

      const int count = (int) other.fields.size();
      for (int i = 0; i < count; ++i) {
        fields.push_back(
          other.fields[i].clone()
        );
      }
    }

    int union_t::type() const {
      return typeType::union_;
    }

    type_t& union_t::clone() const {
      return *(new union_t(*this));
    }

    dtype_t union_t::dtype() const {
      dtype_t dtype_;

      const int fieldCount = (int) fields.size();
      for (int i = 0; i < fieldCount; ++i) {
        const variable_t &var = fields[i];
        dtype_.addField(var.name(),
                        var.dtype());
      }

      return dtype_;
    }

    void union_t::addField(variable_t &field) {
      fields.push_back(field.clone());
    }

    void union_t::addFields(variableVector &fields_) {
      const int fieldCount = (int) fields_.size();
      for (int i = 0; i < fieldCount; ++i) {
        fields.push_back(fields_[i].clone());
      }
    }

    void union_t::printDeclaration(printer &pout) const {
      const std::string name_ = name();
      if (name_.size()) {
        pout << name_ << ' ';
      }

      const int fieldCount = (int) fields.size();
      if (!fieldCount) {
        pout << "{}";
      } else {
        vartype_t prevVartype;

        pout << "{\n";
        pout.addIndentation();
        pout.printIndentation();

        for (int i = 0; i < fieldCount; ++i) {
          const variable_t &var = fields[i];
          if (prevVartype != var.vartype) {
            if (i) {
              pout << ";\n";
              pout.printIndentation();
            }
            prevVartype = var.vartype;
            var.printDeclaration(pout);
          } else {
            pout << ", ";
            var.printExtraDeclaration(pout);
          }
        }
        pout << ";\n";

        pout.removeIndentation();
        pout.printIndentation();
        pout << "}";
      }
    }
  }
}
