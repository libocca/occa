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
#include "parser.hpp"

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

    //---[ Attribute ]------------------
    attributeToken_t::attributeToken_t(const attribute_t &attrType_,
                                       identifierToken &source_) :
      attrType(&attrType_),
      source((identifierToken*) source_.clone()) {}

    attributeToken_t::attributeToken_t(const attributeToken_t &other) {
      *this = other;
    }

    attributeToken_t& attributeToken_t::operator = (const attributeToken_t &other) {
      attrType = other.attrType;
      source   = (identifierToken*) other.source->clone();

      // Copy args
      const int argCount = (int) other.args.size();
      for (int i = 0; i < argCount; ++i) {
        args.push_back(other.args[i]->clone());
      }
      // Copy kwargs
      exprNodeMap::const_iterator it = other.kwargs.begin();
      while (it != other.kwargs.end()) {
        kwargs[it->first] = it->second->clone();
        ++it;
      }

      return *this;
    }

    attributeToken_t::~attributeToken_t() {
      delete source;
      // Free args
      const int argCount = (int) args.size();
      for (int i = 0; i < argCount; ++i) {
        delete args[i];
      }
      // Free kwargs
      exprNodeMap::iterator it = kwargs.begin();
      while (it != kwargs.end()) {
        delete it->second;
        ++it;
      }
    }

    const std::string& attributeToken_t::name() const {
      return source->value;
    }

    bool attributeToken_t::forVariable() const {
      return attrType->forVariable();
    }

    bool attributeToken_t::forFunction() const {
      return attrType->forFunction();
    }

    bool attributeToken_t::forStatement(const int sType) const {
      return attrType->forStatement(sType);
    }

    exprNode* attributeToken_t::operator [] (const int index) {
      if ((0 <= index) && (index < ((int) args.size()))) {
        return args[index];
      }
      return NULL;
    }

    exprNode* attributeToken_t::operator [] (const std::string &arg) {
      exprNodeMap::iterator it = kwargs.find(arg);
      if (it != kwargs.end()) {
        return it->second;
      }
      return NULL;
    }

    void attributeToken_t::printWarning(const std::string &message) {
      source->printWarning(message);
    }

    void attributeToken_t::printError(const std::string &message) {
      source->printError(message);
    }
    //==================================
  }
}
