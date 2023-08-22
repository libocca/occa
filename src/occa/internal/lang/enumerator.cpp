#include <occa/dtype.hpp>
#include <occa/internal/lang/enumerator.hpp>

namespace occa {
  namespace lang {

    enumerator_t::enumerator_t(identifierToken *source_, exprNode *expr_) :
        source((identifierToken*) token_t::clone(source_)),
        expr(exprNode::clone(expr_)) {};

    enumerator_t::enumerator_t(const std::string &name_, exprNode *expr_) :
        source(new identifierToken(fileOrigin(), name_)),
        expr(exprNode::clone(expr_)) {};

    enumerator_t::enumerator_t(const enumerator_t &other) :
        source((identifierToken*) token_t::clone(other.source)),
        expr(other.expr) {};

    enumerator_t& enumerator_t::operator = (const enumerator_t &other) {
      if (this == &other) {
        return *this;
      }
      expr = other.expr;
      if (source != other.source) {
        delete source;
        source = (identifierToken*) token_t::clone(other.source);
      }
      return *this;
    }

    enumerator_t& enumerator_t::clone() const {
      return *(new enumerator_t(*this));
    }

    enumerator_t::~enumerator_t() {}

    void enumerator_t::clear() {
      delete expr;
      expr = NULL;
      delete source;
      source = NULL;
    }

    bool enumerator_t::exists() const {
      return expr;
    }

  }
}