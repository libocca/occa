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
#include "builtins/attributes.hpp"

namespace occa {
  namespace lang {
    //---[ @dim ]-----------------------
    dim::dim() {
    }

    dim::~dim() {
    }

    std::string dim::name() const {
      return "dim";
    }

    attribute_t* dim::create(parser_t &parser,
                             const tokenRangeVector &argRanges) {
      return new dim();
    }

    void dim::onAttributeLoad(parser_t &parser) {
    }

    void dim::beforeStatementLoad(parser_t &parser) {
    }

    void dim::onStatementLoad(parser_t &parser,
                              statement_t &smnt) {
    }
    //==================================

    //---[ @dimOrder ]------------------
    dimOrder::dimOrder() {
    }

    dimOrder::~dimOrder() {
    }

    std::string dimOrder::name() const {
      return "dimOrder";
    }

    attribute_t* dimOrder::create(parser_t &parser,
                                  const tokenRangeVector &argRanges) {
      return new dimOrder();
    }

    void dimOrder::onAttributeLoad(parser_t &parser) {
    }

    void dimOrder::beforeStatementLoad(parser_t &parser) {
    }

    void dimOrder::onStatementLoad(parser_t &parser,
                                   statement_t &smnt) {
    }
    //==================================

    //---[ @tile ]----------------------
    tile::tile() {
    }

    tile::~tile() {
    }

    std::string tile::name() const {
      return "tile";
    }

    attribute_t* tile::create(parser_t &parser,
                              const tokenRangeVector &argRanges) {
      return new tile();
    }

    void tile::onAttributeLoad(parser_t &parser) {
    }

    void tile::beforeStatementLoad(parser_t &parser) {
    }

    void tile::onStatementLoad(parser_t &parser,
                               statement_t &smnt) {
    }
    //==================================

    //---[ @safeTile ]------------------
    safeTile::safeTile() {
    }

    safeTile::~safeTile() {
    }

    std::string safeTile::name() const {
      return "safeTile";
    }

    attribute_t* safeTile::create(parser_t &parser,
                                  const tokenRangeVector &argRanges) {
      return new safeTile();
    }

    void safeTile::onAttributeLoad(parser_t &parser) {
    }

    void safeTile::beforeStatementLoad(parser_t &parser) {
    }

    void safeTile::onStatementLoad(parser_t &parser,
                                   statement_t &smnt) {
    }
    //==================================
  }
}
