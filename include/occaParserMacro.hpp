#ifndef OCCA_PARSER_MACRO_HEADER
#define OCCA_PARSER_MACRO_HEADER

#include "occaParserDefines.hpp"
#include "occaParserTools.hpp"

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
      typeHolder(const std::string strValue, int type_ = noType);

      bool isAFloat() const;

      bool boolValue() const;
      long longValue() const;
      double doubleValue() const;

      void setLongValue(const long &l);
      void setDoubleValue(const double &d);

      operator std::string () const;
    };

    std::ostream& operator << (std::ostream &out, const typeHolder &th);

    int typePrecedence(const typeHolder a, const typeHolder b);

    typeHolder applyOperator(std::string op, const typeHolder a);

    typeHolder applyOperator(const typeHolder a,
                             std::string op,
                             const typeHolder b);

    typeHolder applyOperator(const typeHolder a,
                             std::string op,
                             const typeHolder b,
                             const typeHolder c);
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
