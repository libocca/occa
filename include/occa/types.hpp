/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2017 David Medina and Tim Warburton
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

#ifndef OCCA_TYPES_HEADER
#define OCCA_TYPES_HEADER

#include <iostream>
#include <vector>
#include <map>
#include <stdint.h>

#include "occa/defines.hpp"
#include "occa/vector.hpp"

namespace occa {
  typedef int64_t dim_t;
  typedef uint64_t udim_t;

  typedef std::vector<int>                   intVector_t;
  typedef std::vector<intVector_t>           intVecVector_t;

  typedef std::vector<std::string>           strVector_t;
  typedef strVector_t::iterator              strVectorIterator;
  typedef strVector_t::const_iterator        cStrVectorIterator;

  typedef std::map<std::string, std::string> strToStrMap_t;
  typedef strToStrMap_t::iterator            strToStrMapIterator;
  typedef strToStrMap_t::const_iterator      cStrToStrMapIterator;

  typedef std::map<std::string,strVector_t>  strToStrsMap_t;
  typedef strToStrsMap_t::iterator           strToStrsMapIterator;
  typedef strToStrsMap_t::const_iterator     cStrToStrsMapIterator;

  typedef std::map<std::string, bool>        strToBoolMap_t;
  typedef strToBoolMap_t::iterator           strToBoolMapIterator;
  typedef strToBoolMap_t::const_iterator     cStrToBoolMapIterator;

  //---[ Dim ]--------------------------
  class dim {
  public:
    udim_t x, y, z;

    dim();
    dim(udim_t x_);
    dim(udim_t x_, udim_t y_);
    dim(udim_t x_, udim_t y_, udim_t z_);

    dim(const dim &d);

    dim& operator = (const dim &d);

    dim operator + (const dim &d);
    dim operator - (const dim &d);
    dim operator * (const dim &d);
    dim operator / (const dim &d);

    bool hasNegativeEntries();

    udim_t& operator [] (int i);
    udim_t  operator [] (int i) const;
  };
  //====================================

  //---[ Type To String ]---------------
  template <class TM>
  class typeinfo {
    static const std::string id;
    static const std::string name;
    static const bool isUnsigned;
  };
  //====================================
}

#endif
