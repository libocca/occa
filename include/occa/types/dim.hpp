#ifndef OCCA_TYPES_DIM_HEADER
#define OCCA_TYPES_DIM_HEADER

#include <occa/io/output.hpp>
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

    bool isZero() const;
    bool hasNegativeEntries() const;

    udim_t& operator [] (int i);
    udim_t  operator [] (int i) const;
  };

  std::ostream& operator << (std::ostream &out,
                           const dim &d);
}

#endif
