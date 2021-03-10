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

  bool dim::isZero() const {
    return !(x && y && z);
  }

  bool dim::hasNegativeEntries() const {
    return (
      hasNegativeBitSet(x) ||
      hasNegativeBitSet(y) ||
      hasNegativeBitSet(z)
    );
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
