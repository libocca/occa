#ifndef OCCA_PARSER_MACRO_HEADER
#define OCCA_PARSER_MACRO_HEADER

#include "occaParserDefines.hpp"
#include "occaParserTools.hpp"
#include "occaParserNodes.hpp"

namespace occa {
  namespace parserNS {
    //---[ Op(erator) Holder ]----------------------
    class opHolder {
    public:
      std::string op;
      int type;

      opHolder(const std::string &op_, const int type_);

      bool operator < (const opHolder &h) const;
    };

    typedef std::map<opHolder,int>       opTypeMap_t;
    typedef opTypeMap_t::iterator        opTypeMapIterator;
    typedef opTypeMap_t::const_iterator  cOpTypeMapIterator;

    typedef std::map<std::string,int>    opLevelMap_t;
    typedef opLevelMap_t::iterator       opLevelMapIterator;
    typedef opLevelMap_t::const_iterator cOpLevelMapIterator;

    extern opTypeMap_t opPrecedence;
    extern opLevelMap_t opLevelMap[17];
    extern bool opLevelL2R[17];
    //==============================================


    //---[ Type Holder ]----------------------------
    int toInt(const char *c);
    bool toBool(const char *c);
    char toChar(const char *c);
    long toLong(const char *c);
    short toShort(const char *c);
    float toFloat(const char *c);
    double toDouble(const char *c);

    std::string typeInfoToStr(const int typeInfo);

    class typeHolder {
    public:
      union {
        int int_;
        bool bool_;
        char char_;
        long long_;
        short short_;
        float float_;
        double double_;
      } value;

      int type;

      typeHolder();
      typeHolder(const typeHolder &th);
      typeHolder(const std::string strValue, int type_ = noType);
      typeHolder(const int int__);
      typeHolder(const bool bool__);
      typeHolder(const char char__);
      typeHolder(const long long__);
      typeHolder(const short short__);
      typeHolder(const float float__);
      typeHolder(const double double__);

      typeHolder& operator = (const typeHolder &th);
      typeHolder& operator = (const std::string &str);
      typeHolder& operator = (const int int__);
      typeHolder& operator = (const bool bool__);
      typeHolder& operator = (const char char__);
      typeHolder& operator = (const long long__);
      typeHolder& operator = (const short short__);
      typeHolder& operator = (const float float__);
      typeHolder& operator = (const double double__);

      bool operator == (const typeHolder &th) const;
      bool operator != (const typeHolder &th) const;

      bool operator <  (const typeHolder &th) const;
      bool operator <= (const typeHolder &th) const;
      bool operator >= (const typeHolder &th) const;
      bool operator >  (const typeHolder &th) const;

      bool isAFloat() const;

      bool boolValue() const;
      long longValue() const;
      double doubleValue() const;

      void convertTo(int type_);

      void setLongValue(const long &l);
      void setDoubleValue(const double &d);

      operator std::string () const;
    };

    std::ostream& operator << (std::ostream &out, const typeHolder &th);

    int typePrecedence(typeHolder &a, typeHolder &b);

    typeHolder applyOperator(std::string op, const std::string &a_);
    typeHolder applyOperator(std::string op, typeHolder &a);

    typeHolder applyOperator(const std::string &a_,
                             std::string op,
                             const std::string &b_);

    typeHolder applyOperator(typeHolder &a,
                             std::string op,
                             typeHolder &b);

    typeHolder applyOperator(const std::string &a_,
                             std::string op,
                             const std::string &b_,
                             const std::string &c_);

    typeHolder applyOperator(typeHolder &a,
                             std::string op,
                             typeHolder &b,
                             typeHolder &c);

    typeHolder evaluateString(const std::string &str);
    typeHolder evaluateString(const char *c);

    typeHolder evaluateNode(strNode *nodeRoot);
    //==============================================


    //---[ Macro Info ]-----------------------------
    class macroInfo {
    public:
      std::string name;
      bool isAFunction;

      int argc;
      std::vector<std::string> parts;
      std::vector<int> argBetweenParts;

      macroInfo();

      std::string applyArgs(const std::vector<std::string> &args);
    };

    std::ostream& operator << (std::ostream &out, const macroInfo &info);
    //==============================================
  };
};

#endif
