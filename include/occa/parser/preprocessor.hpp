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

#ifndef OCCA_PARSER_PREPROCESSOR_HEADER
#define OCCA_PARSER_PREPROCESSOR_HEADER

#include "occa/defines.hpp"
#include "occa/tools/sys.hpp"
#include "occa/parser/defines.hpp"
#include "occa/parser/tools.hpp"
#include "occa/parser/nodes.hpp"

namespace occa {
  namespace parserNS {
    //---[ Op(erator) Holder ]----------------------
    class opHolder {
    public:
      std::string op;
      info_t type;

      opHolder(const std::string &op_, const info_t type_);

      bool operator < (const opHolder &h) const;
    };

    typedef std::map<opHolder,info_t>    opTypeMap;
    typedef opTypeMap::iterator          opTypeMapIterator;
    typedef opTypeMap::const_iterator    cOpTypeMapIterator;

    typedef std::map<std::string,info_t> opLevelMap_t;
    typedef opLevelMap_t::iterator       opLevelMapIterator;
    typedef opLevelMap_t::const_iterator cOpLevelMapIterator;

    extern keywordTypeMap cPodTypes;

    extern opTypeMap    *opPrecedence  , cOpPrecedence  , fortranOpPrecedence;
    extern opLevelMap_t *opLevelMap[17], cOpLevelMap[17], fortranOpLevelMap[17];
    extern bool         *opLevelL2R[17], cOpLevelL2R[17], fortranOpLevelL2R[17];

    static const int maxOpLevels = 17;
    //==============================================


    //---[ Type Holder ]----------------------------
    class typeHolder {
    public:
      info_t type;

      // [REFACTOR] Use uint#/int# and map to types based on byte size
      union {
        uintptr_t          void_;

        bool               bool_;
        char               char_;
        unsigned short     ushort_;
        short              short_;
        unsigned int       uint_;
        int                int_;
        unsigned long      ulong_;
        long               long_;
        // unsigned long long ulonglong_;
        // long long          longlong_;
        float              float_;
        double             double_;
      } value;

      typeHolder();
      typeHolder(const typeHolder &th);
      typeHolder(const std::string &str);
      typeHolder(const std::string &str, info_t type_);

      void load(const char *&c);
      void load(const std::string &str);

      typeHolder(const bool bool__);
      typeHolder(const char char__);
      typeHolder(const unsigned short ushort__);
      typeHolder(const short short__);
      typeHolder(const unsigned int uint__);
      typeHolder(const int int__);
      typeHolder(const unsigned long ulong__);
      typeHolder(const long long__);
      // typeHolder(const unsigned long long ulonglong__);
      // typeHolder(const long long longlong__);
      typeHolder(const float float__);
      typeHolder(const double double__);

      typeHolder& operator = (const typeHolder &th);
      typeHolder& operator = (const std::string &str);
      typeHolder& operator = (const bool bool__);
      typeHolder& operator = (const char char__);
      typeHolder& operator = (const unsigned short ushort__);
      typeHolder& operator = (const short short__);
      typeHolder& operator = (const unsigned int uint__);
      typeHolder& operator = (const int int__);
      typeHolder& operator = (const unsigned long ulong__);
      typeHolder& operator = (const long long__);
      // typeHolder& operator = (const unsigned long long ulonglong__);
      // typeHolder& operator = (const long long longlong__);
      typeHolder& operator = (const float float__);
      typeHolder& operator = (const double double__);

      std::string baseTypeStr();
      static std::string typeToBaseTypeStr(info_t type);

      inline info_t maxType(const typeHolder &th1, const typeHolder &th2) {
        return ((th1.type > th2.type) ?
                th1.type              :
                th2.type);
      }

      bool isUnsigned() const;
      bool isAnInt() const;
      bool isALongInt() const;
      // bool isALongLongInt() const;
      bool isAFloat() const;
      bool isADouble() const;

      //   ---[ Unary Operators ]-----------------
      typeHolder  operator ! ();
      typeHolder  operator + ();
      typeHolder  operator - ();
      typeHolder  operator ~ ();
      typeHolder& operator ++ ();
      typeHolder& operator -- ();

      typeHolder operator ++ (int);
      typeHolder operator -- (int);
      //    ======================================


      //   ---[ Boolean Operators ]---------------
      bool operator < (const typeHolder &th);
      bool operator <= (const typeHolder &th);
      bool operator == (const typeHolder &th);
      bool operator != (const typeHolder &th);
      bool operator >= (const typeHolder &th);
      bool operator > (const typeHolder &th);

      bool operator && (const typeHolder &th);
      bool operator || (const typeHolder &th);
      //    ======================================


      //   ---[ Binary Operators ]----------------
      typeHolder operator * (const typeHolder &th);
      typeHolder operator + (const typeHolder &th);
      typeHolder operator - (const typeHolder &th);
      typeHolder operator / (const typeHolder &th);
      typeHolder operator % (const typeHolder &th);

      typeHolder operator & (const typeHolder &th);
      typeHolder operator | (const typeHolder &th);
      typeHolder operator ^ (const typeHolder &th);

      typeHolder operator >> (const typeHolder &th);
      typeHolder operator << (const typeHolder &th);
      //   =======================================


      //   ---[ Assignment Operators ]--------------
      typeHolder operator *= (const typeHolder &th);
      typeHolder operator += (const typeHolder &th);
      typeHolder operator -= (const typeHolder &th);
      typeHolder operator /= (const typeHolder &th);
      typeHolder operator %= (const typeHolder &th);

      typeHolder operator &= (const typeHolder &th);
      typeHolder operator |= (const typeHolder &th);
      typeHolder operator ^= (const typeHolder &th);

      typeHolder operator >>= (const typeHolder &th);
      typeHolder operator <<= (const typeHolder &th);
      //   =======================================

      template <class TM>
      TM to() const {
        switch(type) {
        case boolType      : return (TM) value.bool_;      break;
        case charType      : return (TM) value.char_;      break;
        case ushortType    : return (TM) value.ushort_;    break;
        case shortType     : return (TM) value.short_;     break;
        case uintType      : return (TM) value.uint_;      break;
        case intType       : return (TM) value.int_;       break;
        case ulongType     : return (TM) value.ulong_;     break;
        case longType      : return (TM) value.long_;      break;
          // case ulonglongType : return (TM) value.ulonglong_; break;
          // case longlongType  : return (TM) value.longlong_;  break;
        case floatType     : return (TM) value.float_;     break;
        case doubleType    : return (TM) value.double_;    break;
        default:
          OCCA_ERROR("Value not set\n",
                     false);
          return 0;
        }
      }

      template <class TM>
      void convertFrom(info_t type_) {
        TM oldValue = to<TM>();

        switch(type_) {
        case boolType      : value.bool_       = (bool)               oldValue; break;
        case charType      : value.char_       = (char)               oldValue; break;
        case ushortType    : value.ushort_     = (unsigned short)     oldValue; break;
        case shortType     : value.short_      = (short)              oldValue; break;
        case uintType      : value.uint_       = (unsigned int)       oldValue; break;
        case intType       : value.int_        = (int)                oldValue; break;
        case ulongType     : value.ulong_      = (unsigned long)      oldValue; break;
        case longType      : value.long_       = (long)               oldValue; break;
          // case ulonglongType : value.ulonglong_  = (unsigned long long) oldValue; break;
          // case longlongType  : value.longlong_   = (long long)          oldValue; break;
        case floatType     : value.float_      = (float)              oldValue; break;
        case doubleType    : value.double_     = (double)             oldValue; break;
        default:
          OCCA_ERROR("Value not set\n",
                     false);
        }
      }

      void convertTo(info_t type_);

      template <class TM>
      void set(const TM &t) {
        switch(type) {
        case boolType      : value.bool_       = (bool)                t; break;
        case charType      : value.char_       = (char)                t; break;
        case ushortType    : value.ushort_     = (unsigned short)      t; break;
        case shortType     : value.short_      = (short)               t; break;
        case uintType      : value.uint_       = (unsigned int)        t; break;
        case intType       : value.int_        = (int)                 t; break;
        case ulongType     : value.ulong_      = (unsigned long)       t; break;
        case longType      : value.long_       = (long)                t; break;
          // case ulonglongType : value.ulonglong_  = (unsigned long long)  t; break;
          // case longlongType  : value.longlong_   = (long long)           t; break;
        case floatType     : value.float_      = (float)               t; break;
        case doubleType    : value.double_     = (double)              t; break;
        default:
          OCCA_ERROR("Value not set\n",
                     false);
        }
      }

      operator std::string () const;
    };

    std::ostream& operator << (std::ostream &out, const typeHolder &th);

    info_t typePrecedence(typeHolder &a, typeHolder &b);

    typeHolder applyLOperator(std::string op, const std::string &a_);
    typeHolder applyLOperator(std::string op, typeHolder &a);

    typeHolder applyROperator(const std::string &a_, std::string op);
    typeHolder applyROperator(typeHolder &a, std::string op);

    typeHolder applyLROperator(const std::string &a_,
                               std::string op,
                               const std::string &b_);

    typeHolder applyLROperator(typeHolder &a,
                               std::string op,
                               typeHolder &b);

    typeHolder applyLCROperator(const std::string &a_,
                                std::string op,
                                const std::string &b_,
                                const std::string &c_);

    typeHolder applyLCROperator(typeHolder &a,
                                std::string op,
                                typeHolder &b,
                                typeHolder &c);

    typeHolder evaluateString(const std::string &str, parserBase *parser_ = NULL);
    typeHolder evaluateString(const char *c, parserBase *parser_ = NULL);

    typeHolder evaluateExpression(expNode &expRoot);
    //==============================================


    //---[ Macro Info ]-----------------------------
    class macroInfo {
    public:
      static const int VA_ARGS_POS = -1;

      std::string name;
      bool isAFunction, hasVarArgs;

      int argc;
      std::vector<std::string> parts;
      std::vector<int> argBetweenParts;

      macroInfo();
      void reset();

      std::string applyArgs(const std::vector<std::string> &args);
    };

    std::ostream& operator << (std::ostream &out, const macroInfo &info);
    //==============================================
  }
}

#endif
