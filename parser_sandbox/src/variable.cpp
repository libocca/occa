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
#include "attribute.hpp"
#include "exprNode.hpp"
#include "statement.hpp"
#include "variable.hpp"

namespace occa {
  namespace lang {
    //---[ Variable ]-------------------
    variable_t::variable_t() :
      vartype(),
      source(new identifierToken(filePosition(), "")) {}

    variable_t::variable_t(const vartype_t &vartype_,
                           identifierToken *source_) :
      vartype(vartype_),
      source(NULL) {
      if (source_) {
        source = (identifierToken*) source_->clone();
      } else {
        source = new identifierToken(filePosition(), "");
      }
    }

    variable_t::variable_t(const variable_t &other) :
      vartype(other.vartype),
      source((identifierToken*) other.source->clone()) {
      copyAttributes(attributes, other.attributes);
    }

    variable_t& variable_t::operator = (const variable_t &other) {
      vartype = other.vartype;
      delete source;
      source = (identifierToken*) other.source->clone();

      copyAttributes(attributes, other.attributes);

      return *this;
    }

    variable_t::~variable_t() {
      delete source;
      freeAttributes(attributes);
    }

    bool variable_t::isNamed() const {
      return source->value.size();
    }

    const std::string& variable_t::name() const {
      return source->value;
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

    void variable_t::printDeclaration(printer &pout) const {
      vartype.printDeclaration(pout, name());
    }

    void variable_t::printExtraDeclaration(printer &pout) const {
      vartype.printExtraDeclaration(pout, name());
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
      var(),
      value(NULL) {}

    variableDeclaration::variableDeclaration(const variable_t &var_) :
      var(var_),
      value(NULL) {}

    variableDeclaration::variableDeclaration(const variable_t &var_,
                                             exprNode &value_) :
      var(var_),
      value(&value_) {}

    variableDeclaration::variableDeclaration(const variableDeclaration &other) :
      var(other.var),
      value(NULL) {
      if (other.value) {
        value = &(other.value->clone());
      }
    }

    variableDeclaration::~variableDeclaration() {
      delete value;
    }

    bool variableDeclaration::hasValue() const {
      return value;
    }

    void variableDeclaration::print(printer &pout) const {
      var.printDeclaration(pout);
      if (value) {
        pout << " = " << *value;
      }
    }

    void variableDeclaration::printAsExtra(printer &pout) const {
      var.printExtraDeclaration(pout);
      if (value) {
        pout << " = " << *value;
      }
    }
    //==================================
  }
}
