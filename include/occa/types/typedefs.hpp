#ifndef OCCA_TYPES_TYPEDEFS_HEADER
#define OCCA_TYPES_TYPEDEFS_HEADER

#include <map>
#include <string>
#include <vector>

#include <stdint.h>

namespace occa {
  typedef int64_t dim_t;
  typedef uint64_t udim_t;

  const occa::udim_t UDIM_DEFAULT = static_cast<occa::udim_t>(-1);

  typedef std::vector<int>            intVector;
  typedef std::vector<std::string>    strVector;
  typedef std::map<std::string, bool> strToBoolMap;
}

#endif
