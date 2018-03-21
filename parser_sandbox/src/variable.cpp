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
#include "variable.hpp"
#include "statement.hpp"

namespace occa {
  namespace lang {
    variable::variable(const vartype_t &type_,
                       const std::string &name_) :
      type(type_),
      name(name_) {}

    variable::variable(const variable &other) :
      type(other.type),
      name(other.name) {}

    variable::~variable() {}

    void variable::printDeclaration(printer &pout) const {
      type.printDeclaration(pout, name);
    }

    void variable::printExtraDeclaration(printer &pout) const {
      type.printExtraDeclaration(pout, name);
    }

    printer& operator << (printer &pout,
                          const variable &var) {
      pout << var.name;
      return pout;
    }
  }
}
