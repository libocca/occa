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

#include "occa/parser/preprocessor.hpp"
#include "occa/parser/parser.hpp"
#include "occa/tools/misc.hpp"
#include "occa/tools/string.hpp"

namespace occa {
  namespace parserNS {
    //---[ Op(erator) Holder ]----------------------
    opHolder::opHolder(const std::string &op_, const info_t type_) :
      op(op_),
      type(type_) {}

    bool opHolder::operator < (const opHolder &h) const {
      if (op < h.op)
        return true;
      else if (op > h.op)
        return false;
      else if (type < h.type)
        return true;

      return false;
    }

    keywordTypeMap_t cPodTypes;

    opTypeMap_t  *opPrecedence,   cOpPrecedence,   fortranOpPrecedence;
    opLevelMap_t *opLevelMap[17], cOpLevelMap[17], fortranOpLevelMap[17];
    bool         *opLevelL2R[17], cOpLevelL2R[17], fortranOpLevelL2R[17];
    //==============================================


    //---[ Type Holder ]----------------------------
    typeHolder::typeHolder() {
      type        = noType;
      value.void_ = 0;
    }

    typeHolder::typeHolder(const typeHolder &th) {
      type        = th.type;
      value.void_ = th.value.void_;
    }

    typeHolder::typeHolder(const std::string &str) {
      load(str);
    }

    typeHolder::typeHolder(const std::string &str, info_t type_) {
      if (type_ == noType) {
        load(str);
        return;
      }

      type = type_;

      switch(type) {
      case boolType      : value.bool_       = occa::atoi(str); break;
      case charType      : value.char_       = occa::atoi(str); break;
      case ushortType    : value.ushort_     = occa::atoi(str); break;
      case shortType     : value.short_      = occa::atoi(str); break;
      case uintType      : value.uint_       = occa::atoi(str); break;
      case intType       : value.int_        = occa::atoi(str); break;
      case ulongType     : value.ulong_      = occa::atoi(str); break;
      case longType      : value.long_       = occa::atoi(str); break;
      // case ulonglongType : value.ulonglong_  = occa::atoi(str); break;
      // case longlongType  : value.longlong_   = occa::atoi(str); break;
      case floatType     : value.float_      = occa::atof(str); break;
      case doubleType    : value.double_     = occa::atod(str); break;
      default:
        OCCA_ERROR("Value not set\n",
                   false);
      }
    }

    void typeHolder::load(const char *&c) {
      bool negative  = false;
      bool unsigned_ = false;
      bool decimal   = false;
      bool float_    = false;
      int bits       = 0;
      int longs      = 0;
      int digits     = 0;

      const char *c0 = c;

      if ((strcmp(c, "true")  == 0) ||
         (strcmp(c, "false") == 0)) {

        type        = boolType;
        value.bool_ = (strcmp(c, "true") == 0);

        ++c;
        return;
      }
      else if (strcmp(c, "NULL")  == 0) {
        type        = voidType;
        value.void_ = 0;

        ++c;
        return;
      }

      if ((*c == '+') || (*c == '-')) {
        negative = (*c == '-');
        ++c;
      }

      if (*c == '0') {
        ++digits;
        ++c;

        const char C = upChar(*(c++));

        if (C == 'X')
          bits = 4;
        else if (C == 'B')
          bits = 2;
        else if (('0' <= C) && (C <= '9'))
          bits = 3;
        else
          --c;
      }

      while(true) {
        const char C = upChar(*c);

        if (('0' <= *c) && (*c <= '9')) {
          ++digits;
        }
        else if ((bits == 4) &&
                ('A' <=  C) && ( C <= 'F')) {
          ++digits;
        }
        else if (*c == '.')
          decimal = true;
        else
          break;

        ++c;
      }

      while(*c != '\0') {
        const char C = upChar(*c);

        if (C == 'L') {
          ++longs;
          ++c;
        }
        else if (C == 'U') {
          unsigned_ = true;
          ++c;
        }
        else if (C == 'E') {
          ++c;

          typeHolder exp;
          exp.load(c);

          // Check if there was an 'F' in exp
          float_ = (exp.type == floatType);

          break;
        }
        else if (C == 'F') {
          float_ = true;
          ++c;
        }
        else
          break;
      }

      // If there was something else or no number
      if (((*c != '\0') &&
          !charIsIn(*c, parserNS::cWordDelimiter)) ||
         (digits == 0)) {

        type = noType;

        c = c0;
        return;
      }

      if (decimal || float_) {
        if (!float_) {
          type          = doubleType;
          value.double_ = occa::atod(std::string(c0, c - c0));
        }
        else {
          type         = floatType;
          value.float_ = occa::atof(std::string(c0, c - c0));
        }
      }
      else {
        if (longs == 0) {
          if (!unsigned_) {
            type       = intType;
            value.int_ = occa::atoi(std::string(c0, c - c0));
          }
          else {
            type        = uintType;
            value.uint_ = occa::atoi(std::string(c0, c - c0));
          }
        }
        else if (longs == 1) {
          if (!unsigned_) {
            type        = longType;
            value.long_ = occa::atoi(std::string(c0, c - c0));
          }
          else {
            type         = ulongType;
            value.ulong_ = occa::atoi(std::string(c0, c - c0));
          }
        }
        else {
          type        = voidType;
          value.void_ = occa::atoi(std::string(c0, c - c0));
          // if (!unsigned_) {
          //   type            = longlongType;
          //   value.longlong_ = occa::atoi(std::string(c0, c - c0));
          // }
          // else {
          //   type             = ulonglongType;
          //   value.ulonglong_ = occa::atoi(std::string(c0, c - c0));
          // }
        }
      }

      // Remove warning for not using negative
      ignoreResult(negative);
    }

    void typeHolder::load(const std::string &str) {
      const char *c = str.c_str();
      load(c);
    }

    typeHolder::typeHolder(const bool bool__) {
      type        = boolType;
      value.bool_ = bool__;
    }

    typeHolder::typeHolder(const char char__) {
      type        = charType;
      value.char_ = char__;
    }

    typeHolder::typeHolder(const unsigned short ushort__) {
      type          = ushortType;
      value.ushort_ = ushort__;
    }

    typeHolder::typeHolder(const short short__) {
      type         = shortType;
      value.short_ = short__;
    }

    typeHolder::typeHolder(const unsigned int uint__) {
      type        = uintType;
      value.uint_ = uint__;
    }

    typeHolder::typeHolder(const int int__) {
      type       = intType;
      value.int_ = int__;
    }

    typeHolder::typeHolder(const unsigned long ulong__) {
      type         = ulongType;
      value.ulong_ = ulong__;
    }

    typeHolder::typeHolder(const long long__) {
      type        = longType;
      value.long_ = long__;
    }

    // typeHolder::typeHolder(const unsigned long long ulonglong__) {
    //   type             = ulonglongType;
    //   value.ulonglong_ = ulonglong__;
    // }

    // typeHolder::typeHolder(const long long longlong__) {
    //   type            = longlongType;
    //   value.longlong_ = longlong__;
    // }

    typeHolder::typeHolder(const float float__) {
      type         = floatType;
      value.float_ = float__;
    }

    typeHolder::typeHolder(const double double__) {
      type          = doubleType;
      value.double_ = double__;
    }

    typeHolder& typeHolder::operator = (const typeHolder &th) {
      type        = th.type;
      value.void_ = th.value.void_;

      return *this;
    }

    typeHolder& typeHolder::operator = (const std::string &str) {
      *this = typeHolder(str);

      return *this;
    }

    typeHolder& typeHolder::operator = (const bool bool__) {
      type        = boolType;
      value.bool_ = bool__;

      return *this;
    }

    typeHolder& typeHolder::operator = (const char char__) {
      type        = charType;
      value.char_ = char__;

      return *this;
    }

    typeHolder& typeHolder::operator = (const unsigned short ushort__) {
      type          = ushortType;
      value.ushort_ = ushort__;

      return *this;
    }

    typeHolder& typeHolder::operator = (const short short__) {
      type         = shortType;
      value.short_ = short__;

      return *this;
    }

    typeHolder& typeHolder::operator = (const unsigned int uint__) {
      type        = uintType;
      value.uint_ = uint__;

      return *this;
    }

    typeHolder& typeHolder::operator = (const int int__) {
      type       = intType;
      value.int_ = int__;

      return *this;
    }

    typeHolder& typeHolder::operator = (const unsigned long ulong__) {
      type         = ulongType;
      value.ulong_ = ulong__;

      return *this;
    }

    typeHolder& typeHolder::operator = (const long long__) {
      type        = longType;
      value.long_ = long__;

      return *this;
    }

    // typeHolder& typeHolder::operator = (const unsigned long long ulonglong__) {
    //   type             = ulonglongType;
    //   value.ulonglong_ = ulonglong__;

    //   return *this;
    // }

    // typeHolder& typeHolder::operator = (const long long longlong__) {
    //   type            = longlongType;
    //   value.longlong_ = longlong__;

    //   return *this;
    // }

    typeHolder& typeHolder::operator = (const float float__) {
      type         = floatType;
      value.float_ = float__;

      return *this;
    }

    typeHolder& typeHolder::operator = (const double double__) {
      type          = doubleType;
      value.double_ = double__;

      return *this;
    }

    std::string typeHolder::baseTypeStr() {
      return typeToBaseTypeStr(type);
    }

    std::string typeHolder::typeToBaseTypeStr(info_t type) {
      switch(type) {
      case boolType      : return std::string("bool");   break;
      case charType      : return std::string("char");   break;
      case ushortType    : return std::string("short");  break;
      case shortType     : return std::string("short");  break;
      case uintType      : return std::string("int");    break;
      case intType       : return std::string("int");    break;
      case ulongType     : return std::string("long");   break;
      case longType      : return std::string("long");   break;
      // case ulonglongType : return std::string("long");   break;
      // case longlongType  : return std::string("long");   break;
      case floatType     : return std::string("float");  break;
      case doubleType    : return std::string("double"); break;
      }

      return std::string("N/A");
    }

    bool typeHolder::isUnsigned() const {
      switch(type) {
      case boolType      : return false; break;
      case charType      : return false; break;
      case ushortType    : return true;  break;
      case shortType     : return false; break;
      case uintType      : return true;  break;
      case intType       : return false; break;
      case ulongType     : return true;  break;
      case longType      : return false; break;
      // case ulonglongType : return true;  break;
      // case longlongType  : return false; break;
      case floatType     : return false; break;
      case doubleType    : return false; break;
      default:
        OCCA_ERROR("Value not set\n",
                   false);
        return false;
      }
    }

    bool typeHolder::isAnInt() const {
      switch(type) {
      case boolType      : return true;  break;
      case charType      : return true;  break;
      case ushortType    : return true;  break;
      case shortType     : return true;  break;
      case uintType      : return true;  break;
      case intType       : return true;  break;
      case ulongType     : return true;  break;
      case longType      : return true;  break;
      // case ulonglongType : return true;  break;
      // case longlongType  : return true;  break;
      case floatType     : return false; break;
      case doubleType    : return false; break;
      default:
        OCCA_ERROR("Value not set\n",
                   false);
        return false;
      }
    }

    bool typeHolder::isALongInt() const {
      switch(type) {
      case boolType      : return false; break;
      case charType      : return false; break;
      case ushortType    : return false; break;
      case shortType     : return false; break;
      case uintType      : return false; break;
      case intType       : return false; break;
      case ulongType     : return true;  break;
      case longType      : return true;  break;
      // case ulonglongType : return true;  break;
      // case longlongType  : return true;  break;
      case floatType     : return false; break;
      case doubleType    : return false; break;
      default:
        OCCA_ERROR("Value not set\n",
                   false);
        return false;
      }
    }

    // bool typeHolder::isALongLongInt() const {
    //   switch(type) {
    //   case boolType      : return false; break;
    //   case charType      : return false; break;
    //   case ushortType    : return false; break;
    //   case shortType     : return false; break;
    //   case uintType      : return false; break;
    //   case intType       : return false; break;
    //   case ulongType     : return false; break;
    //   case longType      : return false; break;
    //   case ulonglongType : return true;  break;
    //   case longlongType  : return true;  break;
    //   case floatType     : return false; break;
    //   case doubleType    : return false; break;
    //   default:
    //     OCCA_ERROR(false,
    //                "Value not set\n");
    //     return false;
    //   }
    // }

    bool typeHolder::isAFloat() const {
      switch(type) {
      case boolType      : return false; break;
      case charType      : return false; break;
      case ushortType    : return false; break;
      case shortType     : return false; break;
      case uintType      : return false; break;
      case intType       : return false; break;
      case ulongType     : return false; break;
      case longType      : return false; break;
      // case ulonglongType : return false; break;
      // case longlongType  : return false; break;
      case floatType     : return true;  break;
      case doubleType    : return true;  break;
      default:
        OCCA_ERROR("Value not set\n",
                   false);
        return false;
      }
    }

    bool typeHolder::isADouble() const {
      switch(type) {
      case boolType      : return false; break;
      case charType      : return false; break;
      case ushortType    : return false; break;
      case shortType     : return false; break;
      case uintType      : return false; break;
      case intType       : return false; break;
      case ulongType     : return false; break;
      case longType      : return false; break;
      // case ulonglongType : return false; break;
      // case longlongType  : return false; break;
      case floatType     : return false; break;
      case doubleType    : return true;  break;
      default:
        OCCA_ERROR("Value not set\n",
                   false);
        return false;
      }
    }

    //   ---[ Unary Operators ]-----------------
    typeHolder typeHolder::operator ! () {
      switch(type) {
      case boolType      : return typeHolder(!value.bool_);      break;
      case charType      : return typeHolder(!value.char_);      break;
      case ushortType    : return typeHolder(!value.ushort_);    break;
      case shortType     : return typeHolder(!value.short_);     break;
      case uintType      : return typeHolder(!value.uint_);      break;
      case intType       : return typeHolder(!value.int_);       break;
      case ulongType     : return typeHolder(!value.ulong_);     break;
      case longType      : return typeHolder(!value.long_);      break;
      // case ulonglongType : return typeHolder(!value.ulonglong_); break;
      // case longlongType  : return typeHolder(!value.longlong_);  break;
      case floatType     : return typeHolder(!value.float_);     break;
      case doubleType    : return typeHolder(!value.double_);    break;
      default:
        OCCA_ERROR("Value not set\n",
                   false);
        return typeHolder(0);
      }
    }

    typeHolder typeHolder::operator + () {
      return *this;
    }

    typeHolder typeHolder::operator - () {
      switch(type) {
      case boolType      : return typeHolder(-value.bool_);      break;
      case charType      : return typeHolder(-value.char_);      break;
      case ushortType    : return typeHolder(-value.ushort_);    break;
      case shortType     : return typeHolder(-value.short_);     break;
      case uintType      : return typeHolder(-value.uint_);      break;
      case intType       : return typeHolder(-value.int_);       break;
      case ulongType     : return typeHolder(-value.ulong_);     break;
      case longType      : return typeHolder(-value.long_);      break;
      // case ulonglongType : return typeHolder(-value.ulonglong_); break;
      // case longlongType  : return typeHolder(-value.longlong_);  break;
      case floatType     : return typeHolder(-value.float_);     break;
      case doubleType    : return typeHolder(-value.double_);    break;
      default:
        OCCA_ERROR("Value not set\n",
                   false);
        return typeHolder(0);
      }
    }

    typeHolder typeHolder::operator ~ () {
      switch(type) {
      case boolType      : return typeHolder(~value.bool_);                           break;
      case charType      : return typeHolder(~value.char_);                           break;
      case ushortType    : return typeHolder(~value.ushort_);                         break;
      case shortType     : return typeHolder(~value.short_);                          break;
      case uintType      : return typeHolder(~value.uint_);                           break;
      case intType       : return typeHolder(~value.int_);                            break;
      case ulongType     : return typeHolder(~value.ulong_);                          break;
      case longType      : return typeHolder(~value.long_);                           break;
      // case ulonglongType : return typeHolder(~value.ulonglong_);                      break;
      // case longlongType  : return typeHolder(~value.longlong_);                       break;
      case floatType     : OCCA_ERROR(false, "Can't apply [~] operator to a float");  break;
      case doubleType    : OCCA_ERROR(false, "Can't apply [~] operator to a double"); break;
      default:
        OCCA_ERROR("Value not set\n",
                   false);
      }

      return typeHolder(0);
    }

    typeHolder& typeHolder::operator ++ () {
      switch(type) {
      case boolType      : value.bool_ = !value.bool_; break;
      case charType      : ++value.char_;              break;
      case ushortType    : ++value.ushort_;            break;
      case shortType     : ++value.short_;             break;
      case uintType      : ++value.uint_;              break;
      case intType       : ++value.int_;               break;
      case ulongType     : ++value.ulong_;             break;
      case longType      : ++value.long_;              break;
      // case ulonglongType : ++value.ulonglong_;         break;
      // case longlongType  : ++value.longlong_;          break;
      case floatType     : ++value.float_;             break;
      case doubleType    : ++value.double_;            break;
      default:
        OCCA_ERROR("Value not set\n",
                   false);
      }

      return *this;
    }

    typeHolder& typeHolder::operator -- () {
      switch(type) {
      case boolType      : value.bool_ = !value.bool_; break;
      case charType      : --value.char_;              break;
      case ushortType    : --value.ushort_;            break;
      case shortType     : --value.short_;             break;
      case uintType      : --value.uint_;              break;
      case intType       : --value.int_;               break;
      case ulongType     : --value.ulong_;             break;
      case longType      : --value.long_;              break;
      // case ulonglongType : --value.ulonglong_;         break;
      // case longlongType  : --value.longlong_;          break;
      case floatType     : --value.float_;             break;
      case doubleType    : --value.double_;            break;
      default:
        OCCA_ERROR("Value not set\n",
                   false);
      }

      return *this;
    }

    typeHolder typeHolder::operator ++ (int) {
      switch(type) {
      case boolType      : value.bool_ = !value.bool_; return typeHolder(!value.bool_); break;
      case charType      : return typeHolder(value.char_++);      break;
      case ushortType    : return typeHolder(value.ushort_++);    break;
      case shortType     : return typeHolder(value.short_++);     break;
      case uintType      : return typeHolder(value.uint_++);      break;
      case intType       : return typeHolder(value.int_++);       break;
      case ulongType     : return typeHolder(value.ulong_++);     break;
      case longType      : return typeHolder(value.long_++);      break;
      // case ulonglongType : return typeHolder(value.ulonglong_++); break;
      // case longlongType  : return typeHolder(value.longlong_++);  break;
      case floatType     : return typeHolder(value.float_++);     break;
      case doubleType    : return typeHolder(value.double_++);    break;
      default:
        OCCA_ERROR("Value not set\n",
                   false);
        return typeHolder(0);
      }
    }

    typeHolder typeHolder::operator -- (int) {
      switch(type) {
      case boolType      : value.bool_ = !value.bool_; return typeHolder(!value.bool_); break;
      case charType      : return typeHolder(value.char_--);      break;
      case ushortType    : return typeHolder(value.ushort_--);    break;
      case shortType     : return typeHolder(value.short_--);     break;
      case uintType      : return typeHolder(value.uint_--);      break;
      case intType       : return typeHolder(value.int_--);       break;
      case ulongType     : return typeHolder(value.ulong_--);     break;
      case longType      : return typeHolder(value.long_--);      break;
      // case ulonglongType : return typeHolder(value.ulonglong_--); break;
      // case longlongType  : return typeHolder(value.longlong_--);  break;
      case floatType     : return typeHolder(value.float_--);     break;
      case doubleType    : return typeHolder(value.double_--);    break;
      default:
        OCCA_ERROR("Value not set\n",
                   false);
        return typeHolder(0);
      }
    }
    //    ======================================


    //   ---[ Boolean Operators ]---------------
    bool typeHolder::operator < (const typeHolder &th) {
      const int mtype = typeHolder::maxType(*this, th);

      switch(mtype) {
      case boolType      : return (bool) (to<bool>()               < th.to<bool>());               break;
      case charType      : return (bool) (to<char>()               < th.to<char>());               break;
      case ushortType    : return (bool) (to<unsigned short>()     < th.to<unsigned short>());     break;
      case shortType     : return (bool) (to<short>()              < th.to<short>());              break;
      case uintType      : return (bool) (to<unsigned int>()       < th.to<unsigned int>());       break;
      case intType       : return (bool) (to<int>()                < th.to<int>());                break;
      case ulongType     : return (bool) (to<unsigned long>()      < th.to<unsigned long>());      break;
      case longType      : return (bool) (to<long>()               < th.to<long>());               break;
      // case ulonglongType : return (bool) (to<unsigned long long>() < th.to<unsigned long long>()); break;
      // case longlongType  : return (bool) (to<long long>()          < th.to<long long>());          break;
      case floatType     : return (bool) (to<float>()              < th.to<float>());              break;
      case doubleType    : return (bool) (to<double>()             < th.to<double>());             break;
      default:
        OCCA_ERROR("Value not set\n",
                   false);
        return false;
      }
    }

    bool typeHolder::operator <= (const typeHolder &th) {
      const int mtype = typeHolder::maxType(*this, th);

      switch(mtype) {
      case boolType      : return (bool) (to<bool>()               <= th.to<bool>());               break;
      case charType      : return (bool) (to<char>()               <= th.to<char>());               break;
      case ushortType    : return (bool) (to<unsigned short>()     <= th.to<unsigned short>());     break;
      case shortType     : return (bool) (to<short>()              <= th.to<short>());              break;
      case uintType      : return (bool) (to<unsigned int>()       <= th.to<unsigned int>());       break;
      case intType       : return (bool) (to<int>()                <= th.to<int>());                break;
      case ulongType     : return (bool) (to<unsigned long>()      <= th.to<unsigned long>());      break;
      case longType      : return (bool) (to<long>()               <= th.to<long>());               break;
      // case ulonglongType : return (bool) (to<unsigned long long>() <= th.to<unsigned long long>()); break;
      // case longlongType  : return (bool) (to<long long>()          <= th.to<long long>());          break;
      case floatType     : return (bool) (to<float>()              <= th.to<float>());              break;
      case doubleType    : return (bool) (to<double>()             <= th.to<double>());             break;
      default:
        OCCA_ERROR("Value not set\n",
                   false);
        return false;
      }
    }

    bool typeHolder::operator == (const typeHolder &th) {
      const int mtype = typeHolder::maxType(*this, th);

      switch(mtype) {
      case boolType      : return (bool) (to<bool>()               == th.to<bool>());               break;
      case charType      : return (bool) (to<char>()               == th.to<char>());               break;
      case ushortType    : return (bool) (to<unsigned short>()     == th.to<unsigned short>());     break;
      case shortType     : return (bool) (to<short>()              == th.to<short>());              break;
      case uintType      : return (bool) (to<unsigned int>()       == th.to<unsigned int>());       break;
      case intType       : return (bool) (to<int>()                == th.to<int>());                break;
      case ulongType     : return (bool) (to<unsigned long>()      == th.to<unsigned long>());      break;
      case longType      : return (bool) (to<long>()               == th.to<long>());               break;
      // case ulonglongType : return (bool) (to<unsigned long long>() == th.to<unsigned long long>()); break;
      // case longlongType  : return (bool) (to<long long>()          == th.to<long long>());          break;
      case floatType     : return (bool) (to<float>()              == th.to<float>());              break;
      case doubleType    : return (bool) (to<double>()             == th.to<double>());             break;
      default:
        OCCA_ERROR("Value not set\n",
                   false);
        return false;
      }
    }

    bool typeHolder::operator != (const typeHolder &th) {
      const int mtype = typeHolder::maxType(*this, th);

      switch(mtype) {
      case boolType      : return (bool) (to<bool>()               != th.to<bool>());               break;
      case charType      : return (bool) (to<char>()               != th.to<char>());               break;
      case ushortType    : return (bool) (to<unsigned short>()     != th.to<unsigned short>());     break;
      case shortType     : return (bool) (to<short>()              != th.to<short>());              break;
      case uintType      : return (bool) (to<unsigned int>()       != th.to<unsigned int>());       break;
      case intType       : return (bool) (to<int>()                != th.to<int>());                break;
      case ulongType     : return (bool) (to<unsigned long>()      != th.to<unsigned long>());      break;
      case longType      : return (bool) (to<long>()               != th.to<long>());               break;
      // case ulonglongType : return (bool) (to<unsigned long long>() != th.to<unsigned long long>()); break;
      // case longlongType  : return (bool) (to<long long>()          != th.to<long long>());          break;
      case floatType     : return (bool) (to<float>()              != th.to<float>());              break;
      case doubleType    : return (bool) (to<double>()             != th.to<double>());             break;
      default:
        OCCA_ERROR("Value not set\n",
                   false);
        return false;
      }
    }

    bool typeHolder::operator >= (const typeHolder &th) {
      const int mtype = typeHolder::maxType(*this, th);

      switch(mtype) {
      case boolType      : return (bool) (to<bool>()               >= th.to<bool>());               break;
      case charType      : return (bool) (to<char>()               >= th.to<char>());               break;
      case ushortType    : return (bool) (to<unsigned short>()     >= th.to<unsigned short>());     break;
      case shortType     : return (bool) (to<short>()              >= th.to<short>());              break;
      case uintType      : return (bool) (to<unsigned int>()       >= th.to<unsigned int>());       break;
      case intType       : return (bool) (to<int>()                >= th.to<int>());                break;
      case ulongType     : return (bool) (to<unsigned long>()      >= th.to<unsigned long>());      break;
      case longType      : return (bool) (to<long>()               >= th.to<long>());               break;
      // case ulonglongType : return (bool) (to<unsigned long long>() >= th.to<unsigned long long>()); break;
      // case longlongType  : return (bool) (to<long long>()          >= th.to<long long>());          break;
      case floatType     : return (bool) (to<float>()              >= th.to<float>());              break;
      case doubleType    : return (bool) (to<double>()             >= th.to<double>());             break;
      default:
        OCCA_ERROR("Value not set\n",
                   false);
        return false;
      }
    }

    bool typeHolder::operator > (const typeHolder &th) {
      const int mtype = typeHolder::maxType(*this, th);

      switch(mtype) {
      case boolType      : return (bool) (to<bool>()               > th.to<bool>());               break;
      case charType      : return (bool) (to<char>()               > th.to<char>());               break;
      case ushortType    : return (bool) (to<unsigned short>()     > th.to<unsigned short>());     break;
      case shortType     : return (bool) (to<short>()              > th.to<short>());              break;
      case uintType      : return (bool) (to<unsigned int>()       > th.to<unsigned int>());       break;
      case intType       : return (bool) (to<int>()                > th.to<int>());                break;
      case ulongType     : return (bool) (to<unsigned long>()      > th.to<unsigned long>());      break;
      case longType      : return (bool) (to<long>()               > th.to<long>());               break;
      // case ulonglongType : return (bool) (to<unsigned long long>() > th.to<unsigned long long>()); break;
      // case longlongType  : return (bool) (to<long long>()          > th.to<long long>());          break;
      case floatType     : return (bool) (to<float>()              > th.to<float>());              break;
      case doubleType    : return (bool) (to<double>()             > th.to<double>());             break;
      default:
        OCCA_ERROR("Value not set\n",
                   false);
        return false;
      }
    }

    bool typeHolder::operator && (const typeHolder &th) {
      const int mtype = typeHolder::maxType(*this, th);

      switch(mtype) {
      case boolType      : return (bool) (to<bool>()               && th.to<bool>());               break;
      case charType      : return (bool) (to<char>()               && th.to<char>());               break;
      case ushortType    : return (bool) (to<unsigned short>()     && th.to<unsigned short>());     break;
      case shortType     : return (bool) (to<short>()              && th.to<short>());              break;
      case uintType      : return (bool) (to<unsigned int>()       && th.to<unsigned int>());       break;
      case intType       : return (bool) (to<int>()                && th.to<int>());                break;
      case ulongType     : return (bool) (to<unsigned long>()      && th.to<unsigned long>());      break;
      case longType      : return (bool) (to<long>()               && th.to<long>());               break;
      // case ulonglongType : return (bool) (to<unsigned long long>() && th.to<unsigned long long>()); break;
      // case longlongType  : return (bool) (to<long long>()          && th.to<long long>());          break;
      case floatType     : return (bool) (to<float>()              && th.to<float>());              break;
      case doubleType    : return (bool) (to<double>()             && th.to<double>());             break;
      default:
        OCCA_ERROR("Value not set\n",
                   false);
        return false;
      }
    }

    bool typeHolder::operator || (const typeHolder &th) {
      const int mtype = typeHolder::maxType(*this, th);

      switch(mtype) {
      case boolType      : return (bool) (to<bool>()               || th.to<bool>());               break;
      case charType      : return (bool) (to<char>()               || th.to<char>());               break;
      case ushortType    : return (bool) (to<unsigned short>()     || th.to<unsigned short>());     break;
      case shortType     : return (bool) (to<short>()              || th.to<short>());              break;
      case uintType      : return (bool) (to<unsigned int>()       || th.to<unsigned int>());       break;
      case intType       : return (bool) (to<int>()                || th.to<int>());                break;
      case ulongType     : return (bool) (to<unsigned long>()      || th.to<unsigned long>());      break;
      case longType      : return (bool) (to<long>()               || th.to<long>());               break;
      // case ulonglongType : return (bool) (to<unsigned long long>() || th.to<unsigned long long>()); break;
      // case longlongType  : return (bool) (to<long long>()          || th.to<long long>());          break;
      case floatType     : return (bool) (to<float>()              || th.to<float>());              break;
      case doubleType    : return (bool) (to<double>()             || th.to<double>());             break;
      default:
        OCCA_ERROR("Value not set\n",
                   false);
        return false;
      }
    }
    //    ======================================


    //   ---[ Binary Operators ]----------------
    typeHolder typeHolder::operator * (const typeHolder &th) {
      const int mtype = typeHolder::maxType(*this, th);

      switch(mtype) {
      case boolType      : return typeHolder(to<bool>()               * th.to<bool>());               break;
      case charType      : return typeHolder(to<char>()               * th.to<char>());               break;
      case ushortType    : return typeHolder(to<unsigned short>()     * th.to<unsigned short>());     break;
      case shortType     : return typeHolder(to<short>()              * th.to<short>());              break;
      case uintType      : return typeHolder(to<unsigned int>()       * th.to<unsigned int>());       break;
      case intType       : return typeHolder(to<int>()                * th.to<int>());                break;
      case ulongType     : return typeHolder(to<unsigned long>()      * th.to<unsigned long>());      break;
      case longType      : return typeHolder(to<long>()               * th.to<long>());               break;
      // case ulonglongType : return typeHolder(to<unsigned long long>() * th.to<unsigned long long>()); break;
      // case longlongType  : return typeHolder(to<long long>()          * th.to<long long>());          break;
      case floatType     : return typeHolder(to<float>()              * th.to<float>());              break;
      case doubleType    : return typeHolder(to<double>()             * th.to<double>());             break;
      default:
        OCCA_ERROR("Value not set\n",
                   false);
      }

      return typeHolder(0);
    }

    typeHolder typeHolder::operator + (const typeHolder &th) {
      const int mtype = typeHolder::maxType(*this, th);

      switch(mtype) {
      case boolType      : return typeHolder(to<bool>()               + th.to<bool>());               break;
      case charType      : return typeHolder(to<char>()               + th.to<char>());               break;
      case ushortType    : return typeHolder(to<unsigned short>()     + th.to<unsigned short>());     break;
      case shortType     : return typeHolder(to<short>()              + th.to<short>());              break;
      case uintType      : return typeHolder(to<unsigned int>()       + th.to<unsigned int>());       break;
      case intType       : return typeHolder(to<int>()                + th.to<int>());                break;
      case ulongType     : return typeHolder(to<unsigned long>()      + th.to<unsigned long>());      break;
      case longType      : return typeHolder(to<long>()               + th.to<long>());               break;
      // case ulonglongType : return typeHolder(to<unsigned long long>() + th.to<unsigned long long>()); break;
      // case longlongType  : return typeHolder(to<long long>()          + th.to<long long>());          break;
      case floatType     : return typeHolder(to<float>()              + th.to<float>());              break;
      case doubleType    : return typeHolder(to<double>()             + th.to<double>());             break;
      default:
        OCCA_ERROR("Value not set\n",
                   false);
      }

      return typeHolder(0);
    }

    typeHolder typeHolder::operator - (const typeHolder &th) {
      const int mtype = typeHolder::maxType(*this, th);

      switch(mtype) {
      case boolType      : return typeHolder(to<bool>()               - th.to<bool>());               break;
      case charType      : return typeHolder(to<char>()               - th.to<char>());               break;
      case ushortType    : return typeHolder(to<unsigned short>()     - th.to<unsigned short>());     break;
      case shortType     : return typeHolder(to<short>()              - th.to<short>());              break;
      case uintType      : return typeHolder(to<unsigned int>()       - th.to<unsigned int>());       break;
      case intType       : return typeHolder(to<int>()                - th.to<int>());                break;
      case ulongType     : return typeHolder(to<unsigned long>()      - th.to<unsigned long>());      break;
      case longType      : return typeHolder(to<long>()               - th.to<long>());               break;
      // case ulonglongType : return typeHolder(to<unsigned long long>() - th.to<unsigned long long>()); break;
      // case longlongType  : return typeHolder(to<long long>()          - th.to<long long>());          break;
      case floatType     : return typeHolder(to<float>()              - th.to<float>());              break;
      case doubleType    : return typeHolder(to<double>()             - th.to<double>());             break;
      default:
        OCCA_ERROR("Value not set\n",
                   false);
      }

      return typeHolder(0);
    }

    typeHolder typeHolder::operator / (const typeHolder &th) {
      const int mtype = typeHolder::maxType(*this, th);

      switch(mtype) {
      case boolType      : return typeHolder(to<bool>()               / th.to<bool>());               break;
      case charType      : return typeHolder(to<char>()               / th.to<char>());               break;
      case ushortType    : return typeHolder(to<unsigned short>()     / th.to<unsigned short>());     break;
      case shortType     : return typeHolder(to<short>()              / th.to<short>());              break;
      case uintType      : return typeHolder(to<unsigned int>()       / th.to<unsigned int>());       break;
      case intType       : return typeHolder(to<int>()                / th.to<int>());                break;
      case ulongType     : return typeHolder(to<unsigned long>()      / th.to<unsigned long>());      break;
      case longType      : return typeHolder(to<long>()               / th.to<long>());               break;
      // case ulonglongType : return typeHolder(to<unsigned long long>() / th.to<unsigned long long>()); break;
      // case longlongType  : return typeHolder(to<long long>()          / th.to<long long>());          break;
      case floatType     : return typeHolder(to<float>()              / th.to<float>());              break;
      case doubleType    : return typeHolder(to<double>()             / th.to<double>());             break;
      default:
        OCCA_ERROR("Value not set\n",
                   false);
      }

      return typeHolder(0);
    }
    typeHolder typeHolder::operator % (const typeHolder &th) {
      const int mtype = typeHolder::maxType(*this, th);

      switch(mtype) {
      case boolType      : return typeHolder(to<bool>()               % th.to<bool>());               break;
      case charType      : return typeHolder(to<char>()               % th.to<char>());               break;
      case ushortType    : return typeHolder(to<unsigned short>()     % th.to<unsigned short>());     break;
      case shortType     : return typeHolder(to<short>()              % th.to<short>());              break;
      case uintType      : return typeHolder(to<unsigned int>()       % th.to<unsigned int>());       break;
      case intType       : return typeHolder(to<int>()                % th.to<int>());                break;
      case ulongType     : return typeHolder(to<unsigned long>()      % th.to<unsigned long>());      break;
      case longType      : return typeHolder(to<long>()               % th.to<long>());               break;
      // case ulonglongType : return typeHolder(to<unsigned long long>() % th.to<unsigned long long>()); break;
      // case longlongType  : return typeHolder(to<long long>()          % th.to<long long>());          break;
      case floatType     : OCCA_ERROR(false, "Can't apply [%] operator to a float");                  break;
      case doubleType    : OCCA_ERROR(false, "Can't apply [%] operator to a double");                 break;
      default:
        OCCA_ERROR("Value not set\n",
                   false);
      }

      return typeHolder(0);
    }

    typeHolder typeHolder::operator & (const typeHolder &th) {
      const int mtype = typeHolder::maxType(*this, th);

      switch(mtype) {
      case boolType      : return typeHolder(to<bool>()               & th.to<bool>());               break;
      case charType      : return typeHolder(to<char>()               & th.to<char>());               break;
      case ushortType    : return typeHolder(to<unsigned short>()     & th.to<unsigned short>());     break;
      case shortType     : return typeHolder(to<short>()              & th.to<short>());              break;
      case uintType      : return typeHolder(to<unsigned int>()       & th.to<unsigned int>());       break;
      case intType       : return typeHolder(to<int>()                & th.to<int>());                break;
      case ulongType     : return typeHolder(to<unsigned long>()      & th.to<unsigned long>());      break;
      case longType      : return typeHolder(to<long>()               & th.to<long>());               break;
      // case ulonglongType : return typeHolder(to<unsigned long long>() & th.to<unsigned long long>()); break;
      // case longlongType  : return typeHolder(to<long long>()          & th.to<long long>());          break;
      case floatType     : OCCA_ERROR(false, "Can't apply [&] operator to a float");                  break;
      case doubleType    : OCCA_ERROR(false, "Can't apply [&] operator to a double");                 break;
      default:
        OCCA_ERROR("Value not set\n",
                   false);
      }

      return typeHolder(0);
    }

    typeHolder typeHolder::operator | (const typeHolder &th) {
      const int mtype = typeHolder::maxType(*this, th);

      switch(mtype) {
      case boolType      : return typeHolder(to<bool>()               | th.to<bool>());               break;
      case charType      : return typeHolder(to<char>()               | th.to<char>());               break;
      case ushortType    : return typeHolder(to<unsigned short>()     | th.to<unsigned short>());     break;
      case shortType     : return typeHolder(to<short>()              | th.to<short>());              break;
      case uintType      : return typeHolder(to<unsigned int>()       | th.to<unsigned int>());       break;
      case intType       : return typeHolder(to<int>()                | th.to<int>());                break;
      case ulongType     : return typeHolder(to<unsigned long>()      | th.to<unsigned long>());      break;
      case longType      : return typeHolder(to<long>()               | th.to<long>());               break;
      // case ulonglongType : return typeHolder(to<unsigned long long>() | th.to<unsigned long long>()); break;
      // case longlongType  : return typeHolder(to<long long>()          | th.to<long long>());          break;
      case floatType     : OCCA_ERROR(false, "Can't apply [|] operator to a float");                  break;
      case doubleType    : OCCA_ERROR(false, "Can't apply [|] operator to a double");                 break;
      default:
        OCCA_ERROR("Value not set\n",
                   false);
      }

      return typeHolder(0);
    }

    typeHolder typeHolder::operator ^ (const typeHolder &th) {
      const int mtype = typeHolder::maxType(*this, th);

      switch(mtype) {
      case boolType      : return typeHolder(to<bool>()               ^ th.to<bool>());               break;
      case charType      : return typeHolder(to<char>()               ^ th.to<char>());               break;
      case ushortType    : return typeHolder(to<unsigned short>()     ^ th.to<unsigned short>());     break;
      case shortType     : return typeHolder(to<short>()              ^ th.to<short>());              break;
      case uintType      : return typeHolder(to<unsigned int>()       ^ th.to<unsigned int>());       break;
      case intType       : return typeHolder(to<int>()                ^ th.to<int>());                break;
      case ulongType     : return typeHolder(to<unsigned long>()      ^ th.to<unsigned long>());      break;
      case longType      : return typeHolder(to<long>()               ^ th.to<long>());               break;
      // case ulonglongType : return typeHolder(to<unsigned long long>() ^ th.to<unsigned long long>()); break;
      // case longlongType  : return typeHolder(to<long long>()          ^ th.to<long long>());          break;
      case floatType     : OCCA_ERROR(false, "Can't apply [^] operator to a float");                  break;
      case doubleType    : OCCA_ERROR(false, "Can't apply [^] operator to a double");                 break;
      default:
        OCCA_ERROR("Value not set\n",
                   false);
      }

      return typeHolder(0);
    }

    typeHolder typeHolder::operator >> (const typeHolder &th) {
      const int mtype = typeHolder::maxType(*this, th);

      switch(mtype) {
      case boolType      : return typeHolder(to<bool>()               >> th.to<bool>());               break;
      case charType      : return typeHolder(to<char>()               >> th.to<char>());               break;
      case ushortType    : return typeHolder(to<unsigned short>()     >> th.to<unsigned short>());     break;
      case shortType     : return typeHolder(to<short>()              >> th.to<short>());              break;
      case uintType      : return typeHolder(to<unsigned int>()       >> th.to<unsigned int>());       break;
      case intType       : return typeHolder(to<int>()                >> th.to<int>());                break;
      case ulongType     : return typeHolder(to<unsigned long>()      >> th.to<unsigned long>());      break;
      case longType      : return typeHolder(to<long>()               >> th.to<long>());               break;
      // case ulonglongType : return typeHolder(to<unsigned long long>() >> th.to<unsigned long long>()); break;
      // case longlongType  : return typeHolder(to<long long>()          >> th.to<long long>());          break;
      case floatType     : OCCA_ERROR(false, "Can't apply [>>] operator to a float");                  break;
      case doubleType    : OCCA_ERROR(false, "Can't apply [>>] operator to a double");                 break;
      default:
        OCCA_ERROR("Value not set\n",
                   false);
      }

      return typeHolder(0);
    }

    typeHolder typeHolder::operator << (const typeHolder &th) {
      const int mtype = typeHolder::maxType(*this, th);

      switch(mtype) {
      case boolType      : return typeHolder(to<bool>()               << th.to<bool>());               break;
      case charType      : return typeHolder(to<char>()               << th.to<char>());               break;
      case ushortType    : return typeHolder(to<unsigned short>()     << th.to<unsigned short>());     break;
      case shortType     : return typeHolder(to<short>()              << th.to<short>());              break;
      case uintType      : return typeHolder(to<unsigned int>()       << th.to<unsigned int>());       break;
      case intType       : return typeHolder(to<int>()                << th.to<int>());                break;
      case ulongType     : return typeHolder(to<unsigned long>()      << th.to<unsigned long>());      break;
      case longType      : return typeHolder(to<long>()               << th.to<long>());               break;
      // case ulonglongType : return typeHolder(to<unsigned long long>() << th.to<unsigned long long>()); break;
      // case longlongType  : return typeHolder(to<long long>()          << th.to<long long>());          break;
      case floatType     : OCCA_ERROR(false, "Can't apply [<<] operator to a float");                  break;
      case doubleType    : OCCA_ERROR(false, "Can't apply [<<] operator to a double");                 break;
      default:
        OCCA_ERROR("Value not set\n",
                   false);
      }

      return typeHolder(0);
    }
    //   =======================================


    //   ---[ Assignment Operators ]--------------
    typeHolder typeHolder::operator *= (const typeHolder &th) {
      *this = (*this * th);
      return *this;
    }

    typeHolder typeHolder::operator += (const typeHolder &th) {
      *this = (*this + th);
      return *this;
    }

    typeHolder typeHolder::operator -= (const typeHolder &th) {
      *this = (*this - th);
      return *this;
    }

    typeHolder typeHolder::operator /= (const typeHolder &th) {
      *this = (*this / th);
      return *this;
    }

    typeHolder typeHolder::operator %= (const typeHolder &th) {
      *this = (*this % th);
      return *this;
    }

    typeHolder typeHolder::operator &= (const typeHolder &th) {
      *this = (*this & th);
      return *this;
    }

    typeHolder typeHolder::operator |= (const typeHolder &th) {
      *this = (*this | th);
      return *this;
    }

    typeHolder typeHolder::operator ^= (const typeHolder &th) {
      *this = (*this ^ th);
      return *this;
    }

    typeHolder typeHolder::operator >>= (const typeHolder &th) {
      *this = (*this >> th);
      return *this;
    }

    typeHolder typeHolder::operator <<= (const typeHolder &th) {
      *this = (*this << th);
      return *this;
    }
    //   =======================================

    void typeHolder::convertTo(info_t type_) {
      if (isAnInt()) {
        if (isUnsigned()) {
          // if (isALongLongInt())
          //   convertFrom<unsigned long long>(type_);
          if (isALongInt())
            convertFrom<unsigned long>(type_);
          else
            convertFrom<unsigned int>(type_);
        }
        else {
          if (isALongInt())
            convertFrom<int>(type_);
          else
            convertFrom<int>(type_);
        }
      }
      else if (isAFloat()) {
          if (isADouble())
            convertFrom<double>(type_);
          else
            convertFrom<float>(type_);
      }
      else {
        OCCA_ERROR("Value not set\n",
                   false);
      }
    }

    typeHolder::operator std::string () const {
      std::string str;

      switch(type) {
      case intType   : str = occa::toString(value.int_);    break;
      case boolType  : str = occa::toString(value.bool_);   break;
      case charType  : str = occa::toString(value.char_);   break;
      case longType  : str = occa::toString(value.long_);   break;
      case shortType : str = occa::toString(value.short_);  break;
      case floatType : str = occa::toString(value.float_);  break;
      case doubleType: str = occa::toString(value.double_); break;
      default:
        OCCA_ERROR("Value not set\n",
                   false);
      }

      if ((str.find("inf") != std::string::npos)||
         (str.find("INF") != std::string::npos)) {

        return str;
      }

      // double/float is auto-printed properly from
      //  occa::toString()

      if (type == longType)
        str += 'L';

      return str;
    }

    std::ostream& operator << (std::ostream &out, const typeHolder &th) {
      out << (std::string) th;

      return out;
    }

    info_t typePrecedence(typeHolder &a, typeHolder &b) {
      return ((a.type < b.type) ? b.type : a.type);
    }

    typeHolder applyLOperator(std::string op, const std::string &a_) {
      typeHolder a(a_);
      return applyLOperator(op, a);
    }

    typeHolder applyLOperator(std::string op, typeHolder &a) {
      typeHolder ret;

      if (op == "!")
        ret = !a;
      else if (op == "+")
        ret = a;
      else if (op == "-")
        ret = -a;
      else if (op == "~")
        ret = ~a;
      else if (op == "++")
        ret = ++a;
      else if (op == "--")
        ret = --a;

      return ret;
    }

    typeHolder applyROperator(const std::string &a_, std::string op) {
      typeHolder a(a_);
      return applyROperator(a, op);
    }

    typeHolder applyROperator(typeHolder &a, std::string op) {
      typeHolder ret;

      if (op == "++")
        ret = a++;
      else if (op == "--")
        ret = a--;

      return ret;
    }

    typeHolder applyLROperator(const std::string &a_,
                               std::string op,
                               const std::string &b_) {
      typeHolder a(a_), b(b_);

      return applyLROperator(a, op, b);
    }

    typeHolder applyLROperator(typeHolder &a,
                               std::string op,
                               typeHolder &b) {

      typeHolder ret;

      if (op == "<")
        ret = (a < b);
      else if (op == "<=")
        ret = (a <= b);
      else if (op == "==")
        ret = (a == b);
      else if (op == "!=")
        ret = (a != b);
      else if (op == ">=")
        ret = (a >= b);
      else if (op == ">")
        ret = (a > b);

      else if (op == "&&")
        ret = (a && b);
      else if (op == "||")
        ret = (a || b);

      else if (op == "*")
        ret = (a * b);
      else if (op == "+")
        ret = (a + b);
      else if (op == "-")
        ret = (a - b);
      else if (op == "/")
        ret = (a / b);
      else if (op == "%")
        ret = (a % b);

      else if (op == "&")
        ret = (a & b);
      else if (op == "|")
        ret = (a | b);
      else if (op == "^")
        ret = (a ^ b);

      else if (op == ">>")
        ret = (a >> b);
      else if (op == "<<")
        ret = (a << b);

      else if (op == "*=")
        ret = (a *= b);
      else if (op == "+=")
        ret = (a += b);
      else if (op == "-=")
        ret = (a -= b);
      else if (op == "/=")
        ret = (a /= b);
      else if (op == "%=")
        ret = (a %= b);

      else if (op == "&=")
        ret = (a &= b);
      else if (op == "|=")
        ret = (a |= b);
      else if (op == "^=")
        ret = (a ^= b);

      else if (op == ">>=")
        ret = (a >>= b);
      else if (op == "<<=")
        ret = (a <<= b);

      return ret;
    }

    typeHolder applyLCROperator(const std::string &a_,
                                std::string op,
                                const std::string &b_,
                                const std::string &c_) {
      typeHolder a(a_), b(b_), c(c_);

      return applyLCROperator(a, op, b, c);
    }

    typeHolder applyLCROperator(typeHolder &a,
                                std::string op,
                                typeHolder &b,
                                typeHolder &c) {

      if (a != typeHolder(0))
        return b;
      else
        return c;
    }

    typeHolder evaluateString(const std::string &str, parserBase *parser) {
      return evaluateString(str.c_str(), parser);
    }

    typeHolder evaluateString(const char *c, parserBase *parser) {
      skipWhitespace(c);

      if (*c == '\0')
        return typeHolder("false");

      expNode lineExpNode(c);

      if (parser != NULL)
        parser->applyMacros(lineExpNode.value);

      compressAllWhitespace(lineExpNode.value);

      labelCode(lineExpNode);
      lineExpNode.organizeNode();

      expNode &flatRoot = *(lineExpNode.makeFlatHandle());

      // Check if a variable snuck in
      for (int i = 0; i < flatRoot.leafCount; ++i) {
        if ( !(flatRoot[i].info & (expType::operator_ |
                                  expType::presetValue)) ) {

          expNode::freeFlatHandle(flatRoot);

          return typeHolder("false");
        }
      }

      expNode::freeFlatHandle(flatRoot);

      return evaluateExpression(lineExpNode);
    }

    typeHolder evaluateExpression(expNode &expRoot) {
      if (expRoot.info & expType::presetValue)
        return typeHolder(expRoot.value);

      if (expRoot.info & expType::C) {
        return evaluateExpression(expRoot[0]);
      }
      else if (expRoot.info & expType::L) {
        return applyLOperator(expRoot.value,
                              evaluateExpression(expRoot[0]));
      }
      else if (expRoot.info & expType::R) {
        return applyROperator(evaluateExpression(expRoot[0]),
                              expRoot.value);
      }
      else if (expRoot.info & expType::LR) {
        return applyLROperator(evaluateExpression(expRoot[0]),
                               expRoot.value,
                               evaluateExpression(expRoot[1]));
      }
      else if (expRoot.info & expType::LCR) {
        return applyLCROperator(evaluateExpression(expRoot[0]),
                                expRoot.value,
                                evaluateExpression(expRoot[1]),
                                evaluateExpression(expRoot[2]));
      }
      else if ((expRoot.info == expType::root) &&
              (0 < expRoot.leafCount)) {

        return evaluateExpression(expRoot[0]);
      }

      return typeHolder("0");
    }
    //==============================================


    //---[ Macro Info ]-----------------------------
    macroInfo::macroInfo() {}

    void macroInfo::reset() {
      argc = 0;
      parts.clear();
      argBetweenParts.clear();

      parts.push_back(""); // First part

      isAFunction = false;
      hasVarArgs  = false;
    }

    std::string macroInfo::applyArgs(const std::vector<std::string> &args) {
      const int inputArgc = (int) args.size();

      if ((!hasVarArgs && (argc != inputArgc)) ||
          ( hasVarArgs && (argc  > inputArgc))) {

        std::cout << "Macro [" << name << "]:\n";
        for (size_t i = 0; i < args.size(); ++i)
          std::cout << "    args[" << i << "] = " << args[i] << '\n';

        OCCA_ERROR("Macro [" << name << "] uses [" << argc << (hasVarArgs ? " + ..." : "")
                   << "] argument(s) ([" << args.size() << "] provided)",
                   false);
      }

      const int subs = argBetweenParts.size();

      std::string ret = parts[0];

      for (int i = 0; i < subs; ++i) {
        const int argPos = argBetweenParts[i];

        if (argPos != VA_ARGS_POS) {
          ret += args[argPos];
        }
        else {
          for (int j = argc; j < inputArgc; ++j) {
            ret += args[j];
            if (j < (inputArgc - 1))
              ret += ',';
          }
        }

        ret += parts[i + 1];
      }

      return ret;
    }

    std::ostream& operator << (std::ostream &out, const macroInfo &info) {
      const int subs = info.argBetweenParts.size();

      out << info.name;

      if (info.parts.size()) {
        out << ": " << info.parts[0];
      }
      for (int i = 0; i < subs; ++i) {
        const int argPos = info.argBetweenParts[i];
        if (argPos != macroInfo::VA_ARGS_POS) {
          out << "ARG" << argPos;
        } else {
          out << "__VA_ARGS__";
        }
        out << info.parts[i + 1];
      }

      return out;
    }
    //==============================================
  }
}
