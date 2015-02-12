#include "occaParserMacro.hpp"

namespace occa {
  namespace parserNS {
    //---[ Op(erator) Holder ]----------------------
    opHolder::opHolder(const std::string &op_, const int type_) :
      op(op_),
      type(type_) {}

    bool opHolder::operator < (const opHolder &h) const {
      if(op < h.op)
        return true;
      else if(op > h.op)
        return false;
      else if(type < h.type)
        return true;

      return false;
    }

    opTypeMap_t opPrecedence;
    opLevelMap_t opLevelMap[17];
    //==============================================


    //---[ Type Holder ]----------------------------
    int toInt(const char *c){
      return atoi(c);
    }

    bool toBool(const char *c){
      if(isAnInt(c))
        return (atoi(c) != 0);
      else if(isAFloat(c))
        return (atof(c) != 0);
      else if(strcmp(c, "true") == 0)
        return true;
      else if(strcmp(c, "false") == 0)
        return false;

      OCCA_CHECK(false,
                 "[" << c << "] is not a bool");

      return false;
    }

    char toChar(const char *c){
      return (char) atoi(c);
    }

    long toLong(const char *c){
      return atol(c);
    }

    short toShort(const char *c){
      return (short) atoi(c);
    }

    float toFloat(const char *c){
      return atof(c);
    }

    double toDouble(const char *c){
      return (double) atof(c);
    }

    std::string typeInfoToStr(const int typeInfo){
      std::string ret = "";

      if(typeInfo & heapPointerType){
        if(typeInfo & functionType)
          ret += "* ";
        else
          ret += " *";

        if(typeInfo & constPointerType)
          ret += " const ";
      }
      else if(typeInfo & referenceType){
        ret += " &";
      }
      else
        ret += " ";

      return ret;
    }

    typeHolder::typeHolder(){
      value.double_ = 0;
      type = noType;
    }

    typeHolder::typeHolder(const std::string strValue, int type_){
      if(type_ == noType){
        if( occa::isAnInt(strValue.c_str()) )
          type = longType;
        else if( occa::isAFloat(strValue.c_str()) )
          type = doubleType;
        else if((strValue == "false") || (strValue == "true"))
          type = boolType;
      }
      else
        type = type_;

      switch(type){
      case intType   : value.int_    = toInt(strValue.c_str());    break;
      case boolType  : value.bool_   = toBool(strValue.c_str());   break;
      case charType  : value.char_   = toChar(strValue.c_str());   break;
      case longType  : value.long_   = toLong(strValue.c_str());   break;
      case shortType : value.short_  = toShort(strValue.c_str());  break;
      case floatType : value.float_  = toFloat(strValue.c_str());  break;
      case doubleType: value.double_ = toDouble(strValue.c_str()); break;
      default:
        OCCA_CHECK(false,
                   "Value not set\n");
      }
    }

    bool typeHolder::isAFloat() const {
      switch(type){
      case intType   : return false; break;
      case boolType  : return false; break;
      case charType  : return false; break;
      case longType  : return false; break;
      case shortType : return false; break;
      case floatType : return true;  break;
      case doubleType: return true;  break;
      default:
        OCCA_CHECK(false,
                   "Value not set\n");
        return false;
      }
    }

    bool typeHolder::boolValue() const {
      switch(type){
      case intType   : return (bool) value.int_;    break;
      case boolType  : return (bool) value.bool_;   break;
      case charType  : return (bool) value.char_;   break;
      case longType  : return (bool) value.long_;   break;
      case shortType : return (bool) value.short_;  break;
      case floatType : return (bool) value.float_;  break;
      case doubleType: return (bool) value.double_; break;
      default:
        OCCA_CHECK(false,
                   "Value not set\n");
        return false;
      }
    }

    long typeHolder::longValue() const {
      switch(type){
      case intType   : return (long) value.int_;    break;
      case boolType  : return (long) value.bool_;   break;
      case charType  : return (long) value.char_;   break;
      case longType  : return (long) value.long_;   break;
      case shortType : return (long) value.short_;  break;
      case floatType : return (long) value.float_;  break;
      case doubleType: return (long) value.double_; break;
      default:
        OCCA_CHECK(false,
                   "Value not set\n");
        return 0;
      }
    }

    double typeHolder::doubleValue() const {
      switch(type){
      case intType   : return (double) value.int_;    break;
      case boolType  : return (double) value.bool_;   break;
      case charType  : return (double) value.char_;   break;
      case longType  : return (double) value.long_;   break;
      case shortType : return (double) value.short_;  break;
      case floatType : return (double) value.float_;  break;
      case doubleType: return (double) value.double_; break;
      default:
        OCCA_CHECK(false,
                   "Value not set\n");
        return 0;
      }
    }

    void typeHolder::setLongValue(const long &l){
      switch(type){
      case intType   : value.int_    = (int)    l; break;
      case boolType  : value.bool_   = (bool)   l; break;
      case charType  : value.char_   = (char)   l; break;
      case longType  : value.long_   = (long)   l; break;
      case shortType : value.short_  = (short)  l; break;
      case floatType : value.float_  = (float)  l; break;
      case doubleType: value.double_ = (double) l; break;
      default:
        OCCA_CHECK(false,
                   "Value not set\n");
      }
    }

    void typeHolder::setDoubleValue(const double &d){
      switch(type){
      case intType   : value.int_    = (int)    d; break;
      case boolType  : value.bool_   = (bool)   d; break;
      case charType  : value.char_   = (char)   d; break;
      case longType  : value.long_   = (long)   d; break;
      case shortType : value.short_  = (short)  d; break;
      case floatType : value.float_  = (float)  d; break;
      case doubleType: value.double_ = (double) d; break;
      default:
        OCCA_CHECK(false,
                   "Value not set\n");
      }
    }

    typeHolder::operator std::string () const {
      std::stringstream ss;

      switch(type){
      case intType   : ss << value.int_;    break;
      case boolType  : ss << value.bool_;   break;
      case charType  : ss << value.char_;   break;
      case longType  : ss << value.long_;   break;
      case shortType : ss << value.short_;  break;
      case floatType : ss << value.float_;  break;
      case doubleType: ss << value.double_; break;
      default:
        OCCA_CHECK(false,
                   "Value not set\n");
      }

      return ss.str();
    }

    std::ostream& operator << (std::ostream &out, const typeHolder &th){
      out << (std::string) th;

      return out;
    }

    int typePrecedence(const typeHolder a, const typeHolder b){
      return ((a.type < b.type) ? b.type : a.type);
    }

    typeHolder applyOperator(std::string op, const typeHolder a){
      typeHolder ret;

      if(op == "!"){
        ret.type = boolType;
        if(a.isAFloat())
          ret.setDoubleValue( !a.doubleValue() );
        else
          ret.setLongValue( !a.longValue() );
      }
      else if(op == "+"){
        ret = a;
      }
      else if(op == "-"){
        ret = a;
        if(a.isAFloat())
          ret.setDoubleValue( -a.doubleValue() );
        else
          ret.setLongValue( -a.longValue() );
      }
      else if(op == "~"){
        ret.type = a.type;

        OCCA_CHECK(!a.isAFloat(),
                   "Cannot apply [~] to [" << a << "]");

        ret.setLongValue( ~a.longValue() );
      }

      return ret;
    }

    typeHolder applyOperator(const typeHolder a,
                             std::string op,
                             const typeHolder b){
      typeHolder ret;
      ret.type  = typePrecedence(a,b);
      ret.value = a.value;

      const bool aIsFloat = a.isAFloat();
      const bool bIsFloat = b.isAFloat();

      if(op == "+"){
        if(bIsFloat)
          ret.setDoubleValue(ret.doubleValue() + b.doubleValue());
        else{
          if(aIsFloat)
            ret.setDoubleValue(ret.doubleValue() + b.longValue());
          else
            ret.setLongValue(ret.longValue() + b.longValue());
        }
      }
      else if(op == "-"){
        if(bIsFloat)
          ret.setDoubleValue(ret.doubleValue() - b.doubleValue());
        else{
          if(aIsFloat)
            ret.setDoubleValue(ret.doubleValue() - b.longValue());
          else
            ret.setLongValue(ret.longValue() - b.longValue());
        }
      }
      else if(op == "*"){
        if(bIsFloat)
          ret.setDoubleValue(ret.doubleValue() * b.doubleValue());
        else{
          if(aIsFloat)
            ret.setDoubleValue(ret.doubleValue() * b.longValue());
          else
            ret.setLongValue(ret.longValue() * b.longValue());
        }
      }
      else if(op == "/"){
        if(bIsFloat)
          ret.setDoubleValue(ret.doubleValue() / b.doubleValue());
        else{
          if(aIsFloat)
            ret.setDoubleValue(ret.doubleValue() / b.longValue());
          else
            ret.setLongValue(ret.longValue() / b.longValue());
        }
      }

      else if(op == "+="){
        ret.type = a.type;

        if(bIsFloat)
          ret.setDoubleValue(ret.doubleValue() + b.doubleValue());
        else{
          if(aIsFloat)
            ret.setDoubleValue(ret.doubleValue() + b.longValue());
          else
            ret.setLongValue(ret.longValue() + b.longValue());
        }
      }
      else if(op == "-="){
        ret.type = a.type;

        if(bIsFloat)
          ret.setDoubleValue(ret.doubleValue() - b.doubleValue());
        else{
          if(aIsFloat)
            ret.setDoubleValue(ret.doubleValue() - b.longValue());
          else
            ret.setLongValue(ret.longValue() - b.longValue());
        }
      }
      else if(op == "*="){
        ret.type = a.type;

        if(bIsFloat)
          ret.setDoubleValue(ret.doubleValue() * b.doubleValue());
        else{
          if(aIsFloat)
            ret.setDoubleValue(ret.doubleValue() * b.longValue());
          else
            ret.setLongValue(ret.longValue() * b.longValue());
        }
      }
      else if(op == "/="){
        ret.type = a.type;

        if(bIsFloat)
          ret.setDoubleValue(ret.doubleValue() / b.doubleValue());
        else{
          if(aIsFloat)
            ret.setDoubleValue(ret.doubleValue() / b.longValue());
          else
            ret.setLongValue(ret.longValue() / b.longValue());
        }
      }

      else if(op == "<"){
        ret.type = boolType;

        if(bIsFloat){
          if(aIsFloat)
            ret.setLongValue(a.doubleValue() < b.doubleValue());
          else
            ret.setLongValue(a.longValue() < b.doubleValue());
        }
        else{
          if(aIsFloat)
            ret.setLongValue(a.doubleValue() < b.longValue());
          else
            ret.setLongValue(a.longValue() < b.longValue());
        }
      }
      else if(op == "<="){
        ret.type = boolType;

        if(bIsFloat){
          if(aIsFloat)
            ret.setLongValue(a.doubleValue() <= b.doubleValue());
          else
            ret.setLongValue(a.longValue() <= b.doubleValue());
        }
        else{
          if(aIsFloat)
            ret.setLongValue(a.doubleValue() <= b.longValue());
          else
            ret.setLongValue(a.longValue() <= b.longValue());
        }
      }
      else if(op == "=="){
        ret.type = boolType;

        if(bIsFloat){
          if(aIsFloat)
            ret.setLongValue(a.doubleValue() == b.doubleValue());
          else
            ret.setLongValue(a.longValue() == b.doubleValue());
        }
        else{
          if(aIsFloat)
            ret.setLongValue(a.doubleValue() == b.longValue());
          else
            ret.setLongValue(a.longValue() == b.longValue());
        }
      }
      else if(op == ">="){
        ret.type = boolType;

        if(bIsFloat){
          if(aIsFloat)
            ret.setLongValue(a.doubleValue() >= b.doubleValue());
          else
            ret.setLongValue(a.longValue() >= b.doubleValue());
        }
        else{
          if(aIsFloat)
            ret.setLongValue(a.doubleValue() >= b.longValue());
          else
            ret.setLongValue(a.longValue() >= b.longValue());
        }
      }
      else if(op == ">"){
        ret.type = boolType;

        if(bIsFloat){
          if(aIsFloat)
            ret.setLongValue(a.doubleValue() > b.doubleValue());
          else
            ret.setLongValue(a.longValue() > b.doubleValue());
        }
        else{
          if(aIsFloat)
            ret.setLongValue(a.doubleValue() > b.longValue());
          else
            ret.setLongValue(a.longValue() > b.longValue());
        }
      }
      else if(op == "!="){
        ret.type = boolType;

        if(bIsFloat){
          if(aIsFloat)
            ret.setLongValue(a.doubleValue() != b.doubleValue());
          else
            ret.setLongValue(a.longValue() != b.doubleValue());
        }
        else{
          if(aIsFloat)
            ret.setLongValue(a.doubleValue() != b.longValue());
          else
            ret.setLongValue(a.longValue() != b.longValue());
        }
      }

      else if(op == "="){
        ret.type = a.type;

        if(bIsFloat)
          ret.setDoubleValue( b.doubleValue() );
        else
          ret.setLongValue( b.longValue() );
      }

      else if(op == "<<"){
        OCCA_CHECK(!bIsFloat,
                   "Cannot apply [" << a << " << " << b << "] where " << b << " is [float]");

        OCCA_CHECK(!aIsFloat,
                   "Cannot apply [" << a << " << " << b << "] where " << a << " is [float]");

        ret.setLongValue(ret.longValue() << b.longValue());
      }
      else if(op == ">>"){
        OCCA_CHECK(!bIsFloat,
                   "Cannot apply [" << a << " >> " << b << "] where " << b << " is [float]");

        OCCA_CHECK(!aIsFloat,
                   "Cannot apply [" << a << " >> " << b << "] where " << a << " is [float]");

        ret.setLongValue(ret.longValue() >> b.longValue());
      }
      else if(op == "^"){
        OCCA_CHECK(!bIsFloat,
                   "Cannot apply [" << a << " ^ " << b << "] where " << b << " is [float]");

        OCCA_CHECK(!aIsFloat,
                   "Cannot apply [" << a << " ^ " << b << "] where " << a << " is [float]");

        ret.setLongValue(ret.longValue() ^ b.longValue());
      }
      else if(op == "|"){
        OCCA_CHECK(!bIsFloat,
                   "Cannot apply [" << a << " | " << b << "] where " << b << " is [float]");

        OCCA_CHECK(!aIsFloat,
                   "Cannot apply [" << a << " | " << b << "] where " << a << " is [float]");

        ret.setLongValue(ret.longValue() | b.longValue());
      }
      else if(op == "&"){
        OCCA_CHECK(!bIsFloat,
                   "Cannot apply [" << a << " & " << b << "] where " << b << " is [float]");

        OCCA_CHECK(!aIsFloat,
                   "Cannot apply [" << a << " & " << b << "] where " << a << " is [float]");

        ret.setLongValue(ret.longValue() & b.longValue());
      }
      else if(op == "%"){
        OCCA_CHECK(!bIsFloat,
                   "Cannot apply [" << a << " % " << b << "] where " << b << " is [float]");

        OCCA_CHECK(!aIsFloat,
                   "Cannot apply [" << a << " % " << b << "] where " << a << " is [float]");

        ret.setLongValue(ret.longValue() % b.longValue());
      }

      else if(op == "&&")
        ret.setLongValue( a.boolValue() && b.boolValue() );
      else if(op == "||")
        ret.setLongValue( a.boolValue() || b.boolValue() );

      else if(op == "%="){
        OCCA_CHECK(!bIsFloat,
                   "Cannot apply [" << a << " %= " << b << "] where " << b << " is [float]");

        OCCA_CHECK(!aIsFloat,
                   "Cannot apply [" << a << " %= " << b << "] where " << a << " is [float]");

        ret.setLongValue(ret.longValue() % b.longValue());
      }
      else if(op == "&="){
        OCCA_CHECK(!bIsFloat,
                   "Cannot apply [" << a << " &= " << b << "] where " << b << " is [float]");

        OCCA_CHECK(!aIsFloat,
                   "Cannot apply [" << a << " &= " << b << "] where " << a << " is [float]");

        ret.setLongValue(ret.longValue() & b.longValue());
      }
      else if(op == "^="){
        OCCA_CHECK(!bIsFloat,
                   "Cannot apply [" << a << " ^= " << b << "] where " << b << " is [float]");

        OCCA_CHECK(!aIsFloat,
                   "Cannot apply [" << a << " ^= " << b << "] where " << a << " is [float]");

        ret.setLongValue(ret.longValue() ^ b.longValue());
      }
      else if(op == "|="){
        OCCA_CHECK(!bIsFloat,
                   "Cannot apply [" << a << " |= " << b << "] where " << b << " is [float]");

        OCCA_CHECK(!aIsFloat,
                   "Cannot apply [" << a << " |= " << b << "] where " << a << " is [float]");

        ret.setLongValue(ret.longValue() | b.longValue());
      }
      else if(op == ">>="){
        OCCA_CHECK(!bIsFloat,
                   "Cannot apply [" << a << " >>= " << b << "] where " << b << " is [float]");

        OCCA_CHECK(!aIsFloat,
                   "Cannot apply [" << a << " >>= " << b << "] where " << a << " is [float]");

        ret.setLongValue(ret.longValue() >> b.longValue());
      }
      else if(op == "<<="){
        OCCA_CHECK(!bIsFloat,
                   "Cannot apply [" << a << " <<= " << b << "] where " << b << " is [float]");

        OCCA_CHECK(!aIsFloat,
                   "Cannot apply [" << a << " <<= " << b << "] where " << a << " is [float]");

        ret.setLongValue(ret.longValue() << b.longValue());
      }

      return ret;
    }

    typeHolder applyOperator(const typeHolder a,
                             std::string op,
                             const typeHolder b,
                             const typeHolder c){
      bool pickC;

      if(a.isAFloat())
        pickC = (a.doubleValue() == 0);
      else
        pickC = (a.longValue() == 0);

      if(pickC)
        return c;
      else
        return b;
    }
    //==============================================


    //---[ Macro Info ]-----------------------------
    macroInfo::macroInfo(){};

    std::string macroInfo::applyArgs(const std::vector<std::string> &args){
      if(((size_t) argc) != args.size()){
        std::cout << "Macro [" << name << "]:\n";
        for(size_t i = 0; i < args.size(); ++i)
          std::cout << "    args[" << i << "] = " << args[i] << '\n';

        OCCA_CHECK(false,
                   "Macro [" << name << "] uses [" << argc << "] argument(s) ([" << args.size() << "] provided)");
      }

      const int subs = argBetweenParts.size();

      std::string ret = parts[0];

      for(int i = 0; i < subs; ++i){
        const int argPos = argBetweenParts[i];
        ret += args[argPos];
        ret += parts[i + 1];
      }

      return ret;
    }

    std::ostream& operator << (std::ostream &out, const macroInfo &info){
      const int argc = info.argBetweenParts.size();

      out << info.name << ": " << info.parts[0];

      for(int i = 0; i < argc; ++i){
        const int argPos = info.argBetweenParts[i];
        out << "ARG" << argPos << info.parts[i + 1];
      }

      return out;
    }
    //==============================================
  };
};
