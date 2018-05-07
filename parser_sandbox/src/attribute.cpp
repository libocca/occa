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
#include "parser.hpp"

namespace occa {
  namespace lang {
    attribute_t::attribute_t() :
      source(NULL) {}

    attribute_t::attribute_t(identifierToken &source_) :
      source(&source_) {}

    attribute_t::~attribute_t() {
      delete source;
    }

    bool attribute_t::isVariableAttribute() const {
      return false;
    }

    bool attribute_t::isFunctionAttribute() const {
      return false;
    }

    bool attribute_t::isStatementAttribute(const int stype) const {
      return false;
    }

    bool attribute_t::onVariableLoad(parser_t &parser,
                                     variable_t &var) {
      return false;
    }

    bool attribute_t::onFunctionLoad(parser_t &parser,
                                     function_t &func) {
      return false;
    }

    bool attribute_t::onStatementLoad(parser_t &parser,
                                      statement_t &smnt) {
      return false;
    }

    void attribute_t::onUse(parser_t &parser,
                            statement_t &smnt,
                            exprNode &expr) {}

    void attribute_t::printWarning(const std::string &message) {
      if (source) {
        source->printWarning(message);
      } else {
        occa::printWarning(std::cerr, message);
      }
    }

    void attribute_t::printError(const std::string &message) {
      if (source) {
        source->printError(message);
      } else {
        occa::printError(std::cerr, message);
      }
    }

    void copyAttributes(attributePtrVector &dest,
                        const attributePtrVector &src) {
      freeAttributes(dest);
      const int count = (int) src.size();
      for (int i = 0; i < count; ++i) {
        dest.push_back(src[i]->clone());
      }
    }

    void freeAttributes(attributePtrVector &attributes) {
      const int count = (int) attributes.size();
      for (int i = 0; i < count; ++i) {
        delete attributes[i];
      }
      attributes.clear();
    }
  }
}
