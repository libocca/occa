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
#include <occa/lang/parser.hpp>

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
  }
}
