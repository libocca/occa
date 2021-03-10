#include <occa/internal/lang/attribute.hpp>
#include <occa/internal/lang/expr.hpp>
#include <occa/internal/lang/parser.hpp>

namespace occa {
  namespace lang {
    //---[ Attribute Type ]-------------
    attribute_t::~attribute_t() {}

    bool attribute_t::forVariable() const {
      return false;
    }

    bool attribute_t::forFunction() const {
      return false;
    }
    //==================================

    //---[ Attribute Arg ]--------------
    attributeArg_t::attributeArg_t() :
      expr(NULL) {}

    attributeArg_t::attributeArg_t(exprNode *expr_) :
      expr(expr_) {}

    attributeArg_t::attributeArg_t(exprNode *expr_,
                                   attributeTokenMap attributes_) :
      expr(expr_),
      attributes(attributes_) {}

    attributeArg_t::attributeArg_t(const attributeArg_t &other) :
      expr(other.expr),
      attributes(other.attributes) {}

    attributeArg_t& attributeArg_t::operator = (const attributeArg_t &other) {
      expr = other.expr;
      attributes = other.attributes;
      return *this;
    }

    attributeArg_t::~attributeArg_t() {}

    void attributeArg_t::clear() {
      delete expr;
      expr = NULL;
      attributeTokenMap::iterator it = attributes.begin();
      while (it != attributes.end()) {
        it->second.clear();
        ++it;
      }
      attributes.clear();
    }

    bool attributeArg_t::exists() const {
      return expr;
    }
    //==================================

    //---[ Attribute ]------------------
    attributeToken_t::attributeToken_t() :
      attrType(NULL),
      source(NULL) {}

    attributeToken_t::attributeToken_t(const attribute_t &attrType_,
                                       identifierToken &source_) :
      attrType(&attrType_),
      source((identifierToken*) token_t::clone(&source_)) {}

    attributeToken_t::attributeToken_t(const attributeToken_t &other) :
      attrType(NULL),
      source(NULL) {
      copyFrom(other);
    }

    attributeToken_t& attributeToken_t::operator = (const attributeToken_t &other) {
      clear();
      copyFrom(other);
      return *this;
    }

    attributeToken_t::~attributeToken_t() {
      clear();
    }

    void attributeToken_t::copyFrom(const attributeToken_t &other) {
      // Copying an empty attributeToken
      if (!other.source) {
        return;
      }

      attrType = other.attrType;
      source   = (identifierToken*) token_t::clone(other.source);

      // Copy args
      const int argCount = (int) other.args.size();
      for (int i = 0; i < argCount; ++i) {
        const attributeArg_t &attr = other.args[i];
        args.push_back(
          attributeArg_t(exprNode::clone(attr.expr),
                         attr.attributes)
        );
      }
      // Copy kwargs
      attributeArgMap::const_iterator it = other.kwargs.begin();
      while (it != other.kwargs.end()) {
        const attributeArg_t &attr = it->second;
        kwargs[it->first] = (
          attributeArg_t(exprNode::clone(attr.expr),
                         attr.attributes)
        );
        ++it;
      }
    }

    void attributeToken_t::clear() {
      delete source;
      source = NULL;
      // Free args
      const int argCount = (int) args.size();
      for (int i = 0; i < argCount; ++i) {
        args[i].clear();
      }
      args.clear();
      // Free kwargs
      attributeArgMap::iterator it = kwargs.begin();
      while (it != kwargs.end()) {
        it->second.clear();
        ++it;
      }
      kwargs.clear();
    }

    const std::string& attributeToken_t::name() const {
      return attrType->name();
    }

    bool attributeToken_t::forVariable() const {
      return attrType->forVariable();
    }

    bool attributeToken_t::forFunction() const {
      return attrType->forFunction();
    }

    bool attributeToken_t::forStatementType(const int sType) const {
      return attrType->forStatementType(sType);
    }

    attributeArg_t* attributeToken_t::operator [] (const int index) {
      if ((0 <= index) && (index < ((int) args.size()))) {
        return &(args[index]);
      }
      return NULL;
    }

    attributeArg_t* attributeToken_t::operator [] (const std::string &arg) {
      attributeArgMap::iterator it = kwargs.find(arg);
      if (it != kwargs.end()) {
        return &(it->second);
      }
      return NULL;
    }

    void attributeToken_t::printWarning(const std::string &message) const {
      source->printWarning(message);
    }

    void attributeToken_t::printError(const std::string &message) const {
      source->printError(message);
    }
    //==================================

    io::output& operator << (io::output &out,
                               const attributeArg_t &attr) {
      if (attr.expr) {
        out << *(attr.expr);
        if (attr.attributes.size()) {
          out << ' ';
        }
      }
      out << attr.attributes;
      return out;
    }

    io::output& operator << (io::output &out,
                               const attributeToken_t &attr) {
      out << '@' << attr.name();

      const int argCount = (int) attr.args.size();
      const int kwargCount = (int) attr.kwargs.size();
      if (!argCount && !kwargCount) {
        return out;
      }

      out << '(';
      // args
      for (int i = 0; i < argCount; ++i) {
        out << attr.args[i];
        if ((i < (argCount - 1)) || kwargCount) {
          out << ", ";
        }
      }
      // kwargs
      attributeArgMap::const_iterator it = attr.kwargs.begin();
      while (it != attr.kwargs.end()) {
        out << it->first << '=' << it->second;
        ++it;
        if (it != attr.kwargs.end()) {
          out << ", ";
        }
      }
      out << ')';
      return out;
    }

    io::output& operator << (io::output &out,
                               const attributeTokenMap &attributes) {
      attributeTokenMap::const_iterator it = attributes.begin();
      while (it != attributes.end()) {
        out << it->second << '\n';
        ++it;
      }
      return out;
    }
  }
}
