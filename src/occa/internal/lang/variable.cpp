#include <occa/internal/lang/attribute.hpp>
#include <occa/internal/lang/expr.hpp>
#include <occa/internal/lang/statement.hpp>
#include <occa/internal/lang/variable.hpp>

namespace occa {
  namespace lang {
    //---[ Variable ]-------------------
    variable_t::variable_t() :
      vartype(),
      source(new identifierToken(filePosition(), "")) {}

    variable_t::variable_t(const vartype_t &vartype_,
                           const std::string &name_) :
        vartype(vartype_),
        source(new identifierToken(fileOrigin(), name_)) {}

    variable_t::variable_t(const vartype_t &vartype_,
                           identifierToken *source_) :
        vartype(vartype_),
        source((identifierToken*) token_t::clone(source_)) {}

    variable_t::variable_t(const variable_t &other) :
        vartype(other.vartype),
        source((identifierToken*) token_t::clone(other.source)),
        attributes(other.attributes),
        nameOverride(other.nameOverride) {}

    variable_t& variable_t::operator = (const variable_t &other) {
      if (this == &other) {
        return *this;
      }

      vartype = other.vartype;
      attributes = other.attributes;
      nameOverride = other.nameOverride;

      if (source != other.source) {
        delete source;
        source = (identifierToken*) token_t::clone(other.source);
      }

      return *this;
    }

    variable_t::~variable_t() {
      delete source;
      source = NULL;
    }

    bool variable_t::isNamed() const {
      return !(this->name().empty());
    }

    std::string& variable_t::name() {
      if (!nameOverride.size() && source) {
        return source->value;
      }
      return nameOverride;
    }

    std::string variable_t::name() const {
      if (!nameOverride.size() && source) {
        return source->value;
      }
      return nameOverride;
    }

    void variable_t::setName(const std::string &name_) {
      nameOverride = name_;
    }

    variable_t& variable_t::clone() const {
      return *(new variable_t(*this));
    }

    bool variable_t::operator == (const variable_t &other) const {
      if (this == &other) {
        return true;
      }
      if (name() != other.name()) {
        return false;
      }
      return vartype == other.vartype;
    }

    bool variable_t::hasAttribute(const std::string &attr) const {
      return (attributes.find(attr) != attributes.end());
    }

    void variable_t::addAttribute(attributeToken_t &attr) {
      const std::string attributeName = attr.name();
      if (!hasAttribute(attributeName)) {
        attributes[attributeName] = attr;
      }
    }

    // Qualifiers
    bool variable_t::has(const qualifier_t &qualifier) const {
      return vartype.has(qualifier);
    }

    variable_t& variable_t::operator += (const qualifier_t &qualifier) {
      vartype += qualifier;
      return *this;
    }

    variable_t& variable_t::operator -= (const qualifier_t &qualifier) {
      vartype -= qualifier;
      return *this;
    }

    variable_t& variable_t::operator += (const qualifiers_t &qualifiers) {
      vartype += qualifiers;
      return *this;
    }

    void variable_t::add(const fileOrigin &origin,
                         const qualifier_t &qualifier) {
      vartype.add(origin, qualifier);
    }

    void variable_t::add(const qualifierWithSource &qualifier) {
      vartype.add(qualifier);
    }

    void variable_t::add(const int index,
                         const fileOrigin &origin,
                         const qualifier_t &qualifier) {
      vartype.add(index, origin, qualifier);
    }

    void variable_t::add(const int index,
                         const qualifierWithSource &qualifier) {
      vartype.add(index, qualifier);
    }

    // Pointers
    variable_t& variable_t::operator += (const pointer_t &pointer) {
      vartype += pointer;
      return *this;
    }

    variable_t& variable_t::operator += (const pointerVector &pointers) {
      vartype += pointers;
      return *this;
    }

    variable_t& variable_t::operator += (const array_t &array) {
      vartype += array;
      return *this;
    }

    variable_t& variable_t::operator += (const arrayVector &arrays) {
      vartype += arrays;
      return *this;
    }

    dtype_t variable_t::dtype() const {
      return vartype.dtype();
    }

    void variable_t::debugPrint() const {
      printer pout(io::stderr);

      pout << "Declaration:\n";
      printDeclaration(pout);

      pout << "\nExtra:\n";
      printExtraDeclaration(pout);

      pout << "\nEnd\n";
    }

    void variable_t::printDeclaration(printer &pout,
                                      const vartypePrintType_t printType) const {
      vartype.printDeclaration(pout, name(), printType);
    }

    void variable_t::printExtraDeclaration(printer &pout) const {
      vartype.printExtraDeclaration(pout, name());
    }

    void variable_t::printWarning(const std::string &message) const {
      source->printWarning(message);
    }

    void variable_t::printError(const std::string &message) const {
      source->printError(message);
    }

    printer& operator << (printer &pout,
                          const variable_t &var) {
      pout << var.name();
      return pout;
    }
    //==================================

    //---[ Variable Declaration ]-------
    variableDeclaration::variableDeclaration() :
        varNode(NULL),
        value(NULL) {}

    variableDeclaration::variableDeclaration(variable_t &variable_,
                                             exprNode *value_) :
        varNode(new variableNode(variable_.source,
                                 variable_)),
        value(value_) {}

    variableDeclaration::variableDeclaration(variable_t &variable_,
                                             exprNode &value_) :
        varNode(new variableNode(variable_.source,
                                 variable_)),
        value(&value_) {}

    variableDeclaration::variableDeclaration(const variableDeclaration &other) :
        varNode(NULL),
        value(NULL) {
      *this = other;
    }

    variableDeclaration& variableDeclaration::operator = (const variableDeclaration &other) {
      varNode = (variableNode*) exprNode::clone(other.varNode);
      value = exprNode::clone(other.value);

      return *this;
    }

    variableDeclaration::~variableDeclaration() {
      clear();
    }

    variableDeclaration variableDeclaration::clone() const {
      return variableDeclaration(varNode->value.clone(),
                                 exprNode::clone(value));
    }

    bool variableDeclaration::hasVariable() const {
      return varNode;
    }

    bool variableDeclaration::hasValue() const {
      return value;
    }

    variable_t& variableDeclaration::variable() {
      return varNode->value;
    }

    const variable_t& variableDeclaration::variable() const {
      return varNode->value;
    }

    void variableDeclaration::clear() {
      delete varNode;
      delete value;
      varNode = NULL;
      value = NULL;
    }

    void variableDeclaration::setVariable(variableNode *newVarNode) {
      // No need to update anything
      if (varNode == newVarNode) {
        return;
      }

      delete varNode;
      varNode = newVarNode;
    }

    void variableDeclaration::setVariable(variable_t &variable_) {
      // No need to update anything
      if (varNode && &(varNode->value) == &variable_) {
        return;
      }

      delete varNode;
      varNode = new variableNode(variable_.source,
                                 variable_);
    }

    void variableDeclaration::setValue(exprNode *newValue) {
      // No need to update anything
      if (value == newValue) {
        return;
      }

      delete value;
      value = newValue;
    }

    void variableDeclaration::debugPrint() const {
      printer pout(io::stderr);

      pout << "Declaration:\n";
      print(pout, true);

      pout << "\nExtra:\n";
      printAsExtra(pout);

      pout << "\nEnd\n";
    }

    void variableDeclaration::print(printer &pout,
                                    const bool typeDeclared) const {
      const vartypePrintType_t printType = (
        typeDeclared
        ? vartypePrintType_t::typeDeclaration
        : vartypePrintType_t::type
      );
      variable().printDeclaration(pout, printType);
      if (value) {
        pout << " = " << *value;
      }
    }

    void variableDeclaration::printAsExtra(printer &pout) const {
      variable().printExtraDeclaration(pout);
      if (value) {
        pout << " = " << *value;
      }
    }

    void variableDeclaration::printWarning(const std::string &message) const {
      variable().printWarning(message);
    }

    void variableDeclaration::printError(const std::string &message) const {
      variable().printError(message);
    }
    //==================================
  }
}
