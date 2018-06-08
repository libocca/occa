/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2018 David Medina and Tim Warburton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 */
#include <occa/lang/attribute.hpp>
#include <occa/lang/exprNode.hpp>
#include <occa/lang/statement.hpp>
#include <occa/lang/variable.hpp>

namespace occa {
  namespace lang {
    //---[ Variable ]-------------------
    variable_t::variable_t() :
      vartype(),
      source(new identifierToken(filePosition(), "")) {}

    variable_t::variable_t(const vartype_t &vartype_,
                           identifierToken *source_) :
      vartype(vartype_),
      source((identifierToken*) token_t::clone(source_)) {}

    variable_t::variable_t(const variable_t &other) :
      vartype(other.vartype),
      source((identifierToken*) token_t::clone(other.source)),
      attributes(other.attributes) {}

    variable_t& variable_t::operator = (const variable_t &other) {
      if (this == &other) {
        return *this;
      }

      vartype = other.vartype;
      attributes = other.attributes;

      if (source != other.source) {
        delete source;
        source = (identifierToken*) token_t::clone(other.source);
      }

      return *this;
    }

    variable_t::~variable_t() {
      delete source;
    }

    bool variable_t::isNamed() const {
      return source->value.size();
    }

    std::string& variable_t::name() {
      static std::string noName;
      if (source) {
        return source->value;
      }
      return noName;
    }

    const std::string& variable_t::name() const {
      static std::string noName;
      if (source) {
        return source->value;
      }
      return noName;
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

    void variable_t::printDeclaration(printer &pout) const {
      vartype.printDeclaration(pout, name());
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
      variable(NULL),
      value(NULL) {}

    variableDeclaration::variableDeclaration(variable_t &variable_,
                                             exprNode *value_) :
      variable(&variable_),
      value(value_) {}

    variableDeclaration::variableDeclaration(variable_t &variable_,
                                             exprNode &value_) :
      variable(&variable_),
      value(&value_) {}

    variableDeclaration::variableDeclaration(const variableDeclaration &other) :
      variable(other.variable),
      value(other.value) {}

    variableDeclaration::~variableDeclaration() {}

    variableDeclaration variableDeclaration::clone() const {
      if (!variable) {
        return variableDeclaration();
      }
      if (value) {
        return variableDeclaration(variable->clone(),
                                   *(value->clone()));
      }
      return variableDeclaration(variable->clone());
    }

    void variableDeclaration::clear() {
      // Variable gets deleted in the scope
      delete value;
      variable = NULL;
      value = NULL;
    }

    bool variableDeclaration::hasValue() const {
      return value;
    }

    void variableDeclaration::print(printer &pout) const {
      variable->printDeclaration(pout);
      if (value) {
        pout << " = " << *value;
      }
    }

    void variableDeclaration::printAsExtra(printer &pout) const {
      variable->printExtraDeclaration(pout);
      if (value) {
        pout << " = " << *value;
      }
    }

    void variableDeclaration::printWarning(const std::string &message) const {
      variable->printWarning(message);
    }

    void variableDeclaration::printError(const std::string &message) const {
      variable->printError(message);
    }
    //==================================
  }
}
