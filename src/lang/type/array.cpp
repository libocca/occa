#include <occa/internal/lang/expr.hpp>
#include <occa/internal/lang/type/array.hpp>

namespace occa {
  namespace lang {
    array_t::array_t() :
      start(NULL),
      end(NULL),
      size(NULL) {}

    array_t::array_t(const operatorToken &start_,
                     const operatorToken &end_,
                     exprNode *size_) :
      start((operatorToken*) start_.clone()),
      end((operatorToken*) end_.clone()),
      size(size_) {}

    array_t::array_t(const array_t &other) :
      start(NULL),
      end(NULL),
      size(NULL) {
      if (other.start) {
        start = (operatorToken*) other.start->clone();
      }
      if (other.end) {
        end = (operatorToken*) other.end->clone();
      }
      if (other.size) {
        size = other.size->clone();
      }
    }

    array_t::~array_t() {
      delete start;
      delete end;
      delete size;
    }

    bool array_t::hasSize() const {
      return size;
    }

    bool array_t::canEvaluateSize() const {
      return (size &&
              size->canEvaluate());
    }

    primitive array_t::evaluateSize() const {
      return (size
              ? size->evaluate()
              : primitive());
    }

    void array_t::printWarning(const std::string &message) const {
      start->printWarning(message);
    }

    void array_t::printError(const std::string &message) const {
      start->printError(message);
    }

    io::output& operator << (io::output &out,
                               const array_t &array) {
      printer pout(out);
      pout << array;
      return out;
    }

    printer& operator << (printer &pout,
                          const array_t &array) {
      if (array.size) {
        pout << '['<< (*array.size) << ']';
      } else {
        pout << "[]";
      }
      return pout;
    }
  }
}
