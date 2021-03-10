#include <occa/internal/lang/token.hpp>
#include <occa/internal/lang/type/type.hpp>
#include <occa/internal/lang/variable.hpp>

namespace occa {
  namespace lang {
    namespace typeType {
      const int none      = (1 << 0);

      const int primitive   = (1 << 1);
      const int typedef_    = (1 << 2);
      const int functionPtr = (1 << 3);
      const int function    = (1 << 4);
      const int lambda      = (1 << 5);

      const int class_      = (1 << 6);
      const int struct_     = (1 << 7);
      const int union_      = (1 << 8);
      const int enum_       = (1 << 9);
      const int structure   = (class_  |
                               struct_ |
                               union_  |
                               enum_);
    } // namespace typeType

    namespace classAccess {
      const int private_   = (1 << 0);
      const int protected_ = (1 << 1);
      const int public_    = (1 << 2);
    }

    //---[ Type ]-----------------------
    type_t::type_t(const std::string &name_) :
      source(new identifierToken(fileOrigin(),
                                 name_)) {}

    type_t::type_t(const identifierToken &source_) :
      source((identifierToken*) token_t::clone(&source_)) {}

    type_t::type_t(const type_t &other) :
      source((identifierToken*) token_t::clone(other.source)),
      attributes(other.attributes) {}

    type_t::~type_t() {
      delete source;
    }

    void type_t::setSource(const identifierToken &source_) {
      if (source != &source_) {
        delete source;
        source = (identifierToken*) source_.clone();
      }
    }

    const std::string& type_t::name() const {
      static std::string noName;
      if (source) {
        return source->value;
      }
      return noName;
    }

    bool type_t::isNamed() const {
      if (source) {
        return source->value.size();
      }
      return false;
    }

    bool type_t::isPointerType() const {
      return false;
    }

    bool type_t::operator == (const type_t &other) const {
      if (type() != other.type()) {
        return false;
      }
      if (this == &other) {
        return true;
      }
      return equals(other);
    }

    bool type_t::operator != (const type_t &other) const {
      return !(*this == other);
    }

    bool type_t::equals(const type_t &other) const {
      return false;
    }

    bool type_t::hasAttribute(const std::string &attr) const {
      return (attributes.find(attr) != attributes.end());
    }

    void type_t::printWarning(const std::string &message) const {
      if (!source) {
        occa::printWarning(io::stderr, "[No Token] " + message);
      } else {
        source->printWarning(message);
      }
    }

    void type_t::printError(const std::string &message) const {
      if (!source) {
        occa::printError(io::stderr, "[No Token] " + message);
      } else {
        source->printError(message);
      }
    }

    io::output& operator << (io::output &out,
                               const type_t &type) {
      printer pout(out);
      pout << type;
      return out;
    }

    printer& operator << (printer &pout,
                          const type_t &type) {
      pout << type.name();
      return pout;
    }
  }
}
