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

#ifndef OCCA_TYPES_TYPEDEFS_HEADER
#define OCCA_TYPES_TYPEDEFS_HEADER

#include <iostream>
#include <map>
#include <vector>

#include <stdint.h>

namespace occa {
  typedef int64_t dim_t;
  typedef uint64_t udim_t;

  typedef std::vector<int>                   intVector;
  typedef std::vector<intVector>             intVecVector;

  typedef std::vector<std::string>           strVector;
  typedef strVector::iterator                strVectorIterator;
  typedef strVector::const_iterator          cStrVectorIterator;

  typedef std::map<std::string, std::string> strToStrMap;
  typedef strToStrMap::iterator              strToStrMapIterator;
  typedef strToStrMap::const_iterator        cStrToStrMapIterator;

  typedef std::map<std::string,strVector>    strToStrsMap;
  typedef strToStrsMap::iterator             strToStrsMapIterator;
  typedef strToStrsMap::const_iterator       cStrToStrsMapIterator;

  typedef std::map<std::string, bool>        strToBoolMap;
  typedef strToBoolMap::iterator             strToBoolMapIterator;
  typedef strToBoolMap::const_iterator       cStrToBoolMapIterator;
}

#endif
