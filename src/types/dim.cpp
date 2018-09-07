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

#include <occa/types/dim.hpp>

namespace occa {
  dim::dim() :
    dims(0),
    x(1),
    y(1),
    z(1) {}

  dim::dim(udim_t x_) :
    dims(1),
    x(x_),
    y(1),
    z(1) {}

  dim::dim(udim_t x_, udim_t y_) :
    dims(2),
    x(x_),
    y(y_),
    z(1) {}

  dim::dim(udim_t x_, udim_t y_, udim_t z_) :
    dims(3),
    x(x_),
    y(y_),
    z(z_) {}

  dim::dim(int dims_, udim_t x_, udim_t y_, udim_t z_) :
    dims(dims_),
    x(x_),
    y(y_),
    z(z_) {}

  bool dim::operator == (const dim &d) const {
    return ((dims == d.dims) &&
            (x == d.x) &&
            (y == d.y) &&
            (z == d.z));
  }

  dim dim::operator + (const dim &d) const {
    return dim(dims > d.dims ? dims : d.dims,
               x + d.x,
               y + d.y,
               z + d.z);
  }

  dim dim::operator - (const dim &d) const {
    return dim(dims > d.dims ? dims : d.dims,
               x - d.x,
               y - d.y,
               z - d.z);
  }

  dim dim::operator * (const dim &d) const {
    return dim(dims > d.dims ? dims : d.dims,
               x * d.x,
               y * d.y,
               z * d.z);
  }

  dim dim::operator / (const dim &d) const {
    return dim(dims > d.dims ? dims : d.dims,
               x / d.x,
               y / d.y,
               z / d.z);
  }

  bool dim::hasNegativeEntries() {
    return ((x & (1 << (sizeof(udim_t) - 1))) ||
            (y & (1 << (sizeof(udim_t) - 1))) ||
            (z & (1 << (sizeof(udim_t) - 1))));
  }

  udim_t& dim::operator [] (int i) {
    switch(i) {
    case 0 : return x;
    case 1 : return y;
    default: return z;
    }
  }

  udim_t dim::operator [] (int i) const {
    switch(i) {
    case 0 : return x;
    case 1 : return y;
    default: return z;
    }
  }

  std::ostream& operator << (std::ostream &out,
                             const dim &d) {
    out << '[';
    if (d.dims > 0) {
      out << d.x;
      if (d.dims > 1) {
        out << ", " << d.y;
        if (d.dims > 2) {
          out << ", " << d.z;
        }
      }
    }
    out << ']';
    return out;
  }
}
