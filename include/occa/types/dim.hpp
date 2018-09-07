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

#ifndef OCCA_TYPES_DIM_HEADER
#define OCCA_TYPES_DIM_HEADER

#include <occa/types/typedefs.hpp>

namespace occa {
  class dim {
  public:
    int dims;
    udim_t x, y, z;

    dim();
    dim(udim_t x_);
    dim(udim_t x_, udim_t y_);
    dim(udim_t x_, udim_t y_, udim_t z_);
    dim(int dims_, udim_t x_, udim_t y_, udim_t z_);

    bool operator == (const dim &d) const;

    dim operator + (const dim &d) const;
    dim operator - (const dim &d) const;
    dim operator * (const dim &d) const;
    dim operator / (const dim &d) const;

    bool hasNegativeEntries();

    udim_t& operator [] (int i);
    udim_t  operator [] (int i) const;
  };

  std::ostream& operator << (std::ostream &out,
                             const dim &d);
}

#endif
