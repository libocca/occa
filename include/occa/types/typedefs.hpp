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
