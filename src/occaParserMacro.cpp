#include "occaParserMacro.hpp"
#include "occaParser.hpp"

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
    bool opLevelL2R[17];
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

    typeHolder::typeHolder(const typeHolder &th){
      type = th.type;
      value.double_ = th.value.double_;
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

    typeHolder::typeHolder(const int int__){
      *this = int__;
      type  = intType;
    }

    typeHolder::typeHolder(const bool bool__){
      *this = bool__;
      type  = boolType;
    }

    typeHolder::typeHolder(const char char__){
      *this = char__;
      type  = charType;
    }

    typeHolder::typeHolder(const long long__){
      *this = long__;
      type  = longType;
    }

    typeHolder::typeHolder(const short short__){
      *this = short__;
      type  = shortType;
    }

    typeHolder::typeHolder(const float float__){
      *this = float__;
      type  = floatType;
    }

    typeHolder::typeHolder(const double double__){
      *this = double__;
      type  = doubleType;
    }

    typeHolder& typeHolder::operator = (const typeHolder &th){
      type = th.type;
      value.double_ = th.value.double_;

      return *this;
    }

    typeHolder& typeHolder::operator = (const std::string &str){
      *this = typeHolder(str);

      return *this;
    }

    typeHolder& typeHolder::operator = (const int int__){
      value.int_ = int__;
      type       = intType;

      return *this;
    }

    typeHolder& typeHolder::operator = (const bool bool__){
      value.bool_ = bool__;
      type        = boolType;

      return *this;
    }

    typeHolder& typeHolder::operator = (const char char__){
      value.char_ = char__;
      type        = charType;

      return *this;
    }

    typeHolder& typeHolder::operator = (const long long__){
      value.long_ = long__;
      type        = longType;

      return *this;
    }

    typeHolder& typeHolder::operator = (const short short__){
      value.short_ = short__;
      type        = shortType;

      return *this;
    }

    typeHolder& typeHolder::operator = (const float float__){
      value.float_ = float__;
      type        = floatType;

      return *this;
    }

    typeHolder& typeHolder::operator = (const double double__){
      value.double_ = double__;
      type          = doubleType;

      return *this;
    }


    bool typeHolder::operator == (const typeHolder &th) const {
      int maxType = ((type > th.type) ?
                     th.type          :
                     type);

      switch(maxType){
      case intType   : return (longValue()   == th.longValue()  ); break;
      case boolType  : return (boolValue()   == th.boolValue()  ); break;
      case charType  : return (longValue()   == th.longValue()  ); break;
      case longType  : return (longValue()   == th.longValue()  ); break;
      case shortType : return (longValue()   == th.longValue()  ); break;
      case floatType : return (doubleValue() == th.doubleValue()); break;
      case doubleType: return (doubleValue() == th.doubleValue()); break;
      default:
        OCCA_CHECK(false,
                   "Value not set\n");
        return false;
      }
    }

    bool typeHolder::operator != (const typeHolder &th) const {
      return !(*this == th);
    }

    bool typeHolder::operator < (const typeHolder &th) const {
      int maxType = ((type > th.type) ?
                     th.type          :
                     type);

      switch(maxType){
      case intType   : return (longValue()   < th.longValue()  ); break;
      case boolType  : return (boolValue()   < th.boolValue()  ); break;
      case charType  : return (longValue()   < th.longValue()  ); break;
      case longType  : return (longValue()   < th.longValue()  ); break;
      case shortType : return (longValue()   < th.longValue()  ); break;
      case floatType : return (doubleValue() < th.doubleValue()); break;
      case doubleType: return (doubleValue() < th.doubleValue()); break;
      default:
        OCCA_CHECK(false,
                   "Value not set\n");
        return false;
      }
    }

    bool typeHolder::operator <= (const typeHolder &th) const {
      int maxType = ((type > th.type) ?
                     th.type          :
                     type);

      switch(maxType){
      case intType   : return (longValue()   <= th.longValue()  ); break;
      case boolType  : return (boolValue()   <= th.boolValue()  ); break;
      case charType  : return (longValue()   <= th.longValue()  ); break;
      case longType  : return (longValue()   <= th.longValue()  ); break;
      case shortType : return (longValue()   <= th.longValue()  ); break;
      case floatType : return (doubleValue() <= th.doubleValue()); break;
      case doubleType: return (doubleValue() <= th.doubleValue()); break;
      default:
        OCCA_CHECK(false,
                   "Value not set\n");
        return false;
      }
    }

    bool typeHolder::operator >= (const typeHolder &th) const {
      return !(*this < th);
    }

    bool typeHolder::operator > (const typeHolder &th) const {
      return !(*this <= th);
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

    void typeHolder::convertTo(int type_){
      if(isAFloat()){
        double oldValue = doubleValue();

        switch(type_){
        case intType   : value.int_    = (int)    oldValue; break;
        case boolType  : value.bool_   = (bool)   oldValue; break;
        case charType  : value.char_   = (char)   oldValue; break;
        case longType  : value.long_   = (long)   oldValue; break;
        case shortType : value.short_  = (short)  oldValue; break;
        case floatType : value.float_  = (float)  oldValue; break;
        case doubleType: value.double_ = (double) oldValue; break;
        default:
          OCCA_CHECK(false,
                     "Value not set\n");
        }
      }
      else{
        long oldValue = longValue();

        switch(type_){
        case intType   : value.int_    = (int)    oldValue; break;
        case boolType  : value.bool_   = (bool)   oldValue; break;
        case charType  : value.char_   = (char)   oldValue; break;
        case longType  : value.long_   = (long)   oldValue; break;
        case shortType : value.short_  = (short)  oldValue; break;
        case floatType : value.float_  = (float)  oldValue; break;
        case doubleType: value.double_ = (double) oldValue; break;
        default:
          OCCA_CHECK(false,
                     "Value not set\n");
        }
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

    int typePrecedence(typeHolder &a, typeHolder &b){
      return ((a.type < b.type) ? b.type : a.type);
    }

    typeHolder applyOperator(std::string op, const std::string &a_){
      typeHolder a(a_);
      return applyOperator(op, a);
    }

    typeHolder applyOperator(std::string op, typeHolder &a){
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

    typeHolder applyOperator(const std::string &a_,
                             std::string op,
                             const std::string &b_){
      typeHolder a(a_), b(b_);

      return applyOperator(a, op, b);
    }

    typeHolder applyOperator(typeHolder &a,
                             std::string op,
                             typeHolder &b){

      typeHolder ret;
      ret.type = typePrecedence(a,b);

      const bool aIsFloat = a.isAFloat();
      const bool bIsFloat = b.isAFloat();

      if(op == "+"){
        if(bIsFloat)
          ret.setDoubleValue(a.doubleValue() + b.doubleValue());
        else{
          if(aIsFloat)
            ret.setDoubleValue(a.doubleValue() + b.longValue());
          else
            ret.setLongValue(a.longValue() + b.longValue());
        }
      }
      else if(op == "-"){
        if(bIsFloat)
          ret.setDoubleValue(a.doubleValue() - b.doubleValue());
        else{
          if(aIsFloat)
            ret.setDoubleValue(a.doubleValue() - b.longValue());
          else
            ret.setLongValue(a.longValue() - b.longValue());
        }
      }
      else if(op == "*"){
        if(bIsFloat)
          ret.setDoubleValue(a.doubleValue() * b.doubleValue());
        else{
          if(aIsFloat)
            ret.setDoubleValue(a.doubleValue() * b.longValue());
          else
            ret.setLongValue(a.longValue() * b.longValue());
        }
      }
      else if(op == "/"){
        if(bIsFloat)
          ret.setDoubleValue(a.doubleValue() / b.doubleValue());
        else{
          if(aIsFloat)
            ret.setDoubleValue(a.doubleValue() / b.longValue());
          else
            ret.setLongValue(a.longValue() / b.longValue());
        }
      }

      else if(op == "+="){
        ret.type = a.type;

        if(bIsFloat)
          ret.setDoubleValue(a.doubleValue() + b.doubleValue());
        else{
          if(aIsFloat)
            ret.setDoubleValue(a.doubleValue() + b.longValue());
          else
            ret.setLongValue(a.longValue() + b.longValue());
        }
      }
      else if(op == "-="){
        ret.type = a.type;

        if(bIsFloat)
          ret.setDoubleValue(a.doubleValue() - b.doubleValue());
        else{
          if(aIsFloat)
            ret.setDoubleValue(a.doubleValue() - b.longValue());
          else
            ret.setLongValue(a.longValue() - b.longValue());
        }
      }
      else if(op == "*="){
        ret.type = a.type;

        if(bIsFloat)
          ret.setDoubleValue(a.doubleValue() * b.doubleValue());
        else{
          if(aIsFloat)
            ret.setDoubleValue(a.doubleValue() * b.longValue());
          else
            ret.setLongValue(a.longValue() * b.longValue());
        }
      }
      else if(op == "/="){
        ret.type = a.type;

        if(bIsFloat)
          ret.setDoubleValue(a.doubleValue() / b.doubleValue());
        else{
          if(aIsFloat)
            ret.setDoubleValue(a.doubleValue() / b.longValue());
          else
            ret.setLongValue(a.longValue() / b.longValue());
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

        ret.setLongValue(a.longValue() << b.longValue());
      }
      else if(op == ">>"){
        OCCA_CHECK(!bIsFloat,
                   "Cannot apply [" << a << " >> " << b << "] where " << b << " is [float]");

        OCCA_CHECK(!aIsFloat,
                   "Cannot apply [" << a << " >> " << b << "] where " << a << " is [float]");

        ret.setLongValue(a.longValue() >> b.longValue());
      }
      else if(op == "^"){
        OCCA_CHECK(!bIsFloat,
                   "Cannot apply [" << a << " ^ " << b << "] where " << b << " is [float]");

        OCCA_CHECK(!aIsFloat,
                   "Cannot apply [" << a << " ^ " << b << "] where " << a << " is [float]");

        ret.setLongValue(a.longValue() ^ b.longValue());
      }
      else if(op == "|"){
        OCCA_CHECK(!bIsFloat,
                   "Cannot apply [" << a << " | " << b << "] where " << b << " is [float]");

        OCCA_CHECK(!aIsFloat,
                   "Cannot apply [" << a << " | " << b << "] where " << a << " is [float]");

        ret.setLongValue(a.longValue() | b.longValue());
      }
      else if(op == "&"){
        OCCA_CHECK(!bIsFloat,
                   "Cannot apply [" << a << " & " << b << "] where " << b << " is [float]");

        OCCA_CHECK(!aIsFloat,
                   "Cannot apply [" << a << " & " << b << "] where " << a << " is [float]");

        ret.setLongValue(a.longValue() & b.longValue());
      }
      else if(op == "%"){
        OCCA_CHECK(!bIsFloat,
                   "Cannot apply [" << a << " % " << b << "] where " << b << " is [float]");

        OCCA_CHECK(!aIsFloat,
                   "Cannot apply [" << a << " % " << b << "] where " << a << " is [float]");

        ret.setLongValue(a.longValue() % b.longValue());
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

        ret.setLongValue(a.longValue() % b.longValue());
      }
      else if(op == "&="){
        OCCA_CHECK(!bIsFloat,
                   "Cannot apply [" << a << " &= " << b << "] where " << b << " is [float]");

        OCCA_CHECK(!aIsFloat,
                   "Cannot apply [" << a << " &= " << b << "] where " << a << " is [float]");

        ret.setLongValue(a.longValue() & b.longValue());
      }
      else if(op == "^="){
        OCCA_CHECK(!bIsFloat,
                   "Cannot apply [" << a << " ^= " << b << "] where " << b << " is [float]");

        OCCA_CHECK(!aIsFloat,
                   "Cannot apply [" << a << " ^= " << b << "] where " << a << " is [float]");

        ret.setLongValue(a.longValue() ^ b.longValue());
      }
      else if(op == "|="){
        OCCA_CHECK(!bIsFloat,
                   "Cannot apply [" << a << " |= " << b << "] where " << b << " is [float]");

        OCCA_CHECK(!aIsFloat,
                   "Cannot apply [" << a << " |= " << b << "] where " << a << " is [float]");

        ret.setLongValue(a.longValue() | b.longValue());
      }
      else if(op == ">>="){
        OCCA_CHECK(!bIsFloat,
                   "Cannot apply [" << a << " >>= " << b << "] where " << b << " is [float]");

        OCCA_CHECK(!aIsFloat,
                   "Cannot apply [" << a << " >>= " << b << "] where " << a << " is [float]");

        ret.setLongValue(a.longValue() >> b.longValue());
      }
      else if(op == "<<="){
        OCCA_CHECK(!bIsFloat,
                   "Cannot apply [" << a << " <<= " << b << "] where " << b << " is [float]");

        OCCA_CHECK(!aIsFloat,
                   "Cannot apply [" << a << " <<= " << b << "] where " << a << " is [float]");

        ret.setLongValue(a.longValue() << b.longValue());
      }

      return ret;
    }

    typeHolder applyOperator(const std::string &a_,
                             std::string op,
                             const std::string &b_,
                             const std::string &c_){
      typeHolder a(a_), b(b_), c(c_);

      return applyOperator(a, op, b, c);
    }

    typeHolder applyOperator(typeHolder &a,
                             std::string op,
                             typeHolder &b,
                             typeHolder &c){
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

    typeHolder evaluateString(const std::string &str){
      return evaluateString(str.c_str());
    }

    typeHolder evaluateString(const char *c){
      skipWhitespace(c);

      if(*c == '\0')
        return typeHolder("false");

      strNode *lineNode = new strNode(c);
      strip(lineNode->value);

      strNode *labelNodeRoot = labelCode(lineNode);
      strNode *labelNodePos  = labelNodeRoot;

      // Check if a variable snuck in
      while(labelNodePos){
        if(labelNodePos->info & unknownVariable)
          return typeHolder("false");

        labelNodePos = labelNodePos->right;
      }

      return evaluateNode(labelNodeRoot);
    }

    typeHolder evaluateNode(strNode *nodeRoot){
      if(nodeRoot == NULL)
        return typeHolder("0");

      if((nodeRoot->info  & presetValue) &&
         (nodeRoot->right == NULL)){

        return typeHolder(*nodeRoot);
      }

      strNode *labelNodePos = nodeRoot;

      while(labelNodePos){
        if(labelNodePos->down){
          labelNodePos->value = evaluateNode(labelNodePos->down);
          labelNodePos->info  = presetValue;
        }

        if(labelNodePos->right == NULL)
          break;

        labelNodePos = labelNodePos->right;
      }

      strNode *minOpNode;
      int minPrecedence, minOpType;

      labelNodePos = nodeRoot;

      while(true){
        minOpNode     = NULL;
        minPrecedence = 100;
        minOpType     = -1;

        while(labelNodePos){
          if(labelNodePos->info & operatorType){
            int opType = (labelNodePos->info & operatorType);

            opType &= ~qualifierType;

            if(opType & unitaryOperatorType){
              if((opType & binaryOperatorType) && // + and - operators
                 (labelNodePos->left)          &&
                 (labelNodePos->left->info & presetValue)){

                opType = binaryOperatorType;
              }
              else if((opType & rUnitaryOperatorType) &&
                      (labelNodePos->left)            &&
                      (labelNodePos->left->info & presetValue)){

                opType = rUnitaryOperatorType;
              }
              else if((opType & lUnitaryOperatorType) &&
                      (labelNodePos->right)           &&
                      (labelNodePos->right->info & presetValue)){

                opType = lUnitaryOperatorType;
              }
              else
                opType &= ~unitaryOperatorType;
            }

            const int opP = opPrecedence[opHolder(labelNodePos->value,
                                                  opType)];

            if(opP < minPrecedence){
              minOpType     = opType;
              minOpNode     = labelNodePos;
              minPrecedence = opP;
            }
          }

          labelNodePos = labelNodePos->right;
        }

        if(minOpNode == NULL){
          if(nodeRoot && (nodeRoot->right == NULL))
            return typeHolder(*nodeRoot);

          OCCA_CHECK(false,
                     "5. Error on:\n"
                     << *(nodeRoot));
        }
        else{
          if(minOpType & unitaryOperatorType){
            if(minOpType & lUnitaryOperatorType){
              std::string op = minOpNode->value;
              std::string a  = minOpNode->right->value;

              minOpNode->value = applyOperator(op, a);
              minOpNode->info  = presetValue;

              minOpNode->right->pop();
            }
            else if(minOpType & rUnitaryOperatorType){
              OCCA_CHECK(false,
                         "Postfix operator [" << *minOpNode << "] cannot be used in a macro");
            }
          }
          else if(minOpType & binaryOperatorType){
            minOpNode = minOpNode->left;

            std::string a  = minOpNode->value;
            std::string op = minOpNode->right->value;
            std::string b  = minOpNode->right->right->value;

            minOpNode->value = applyOperator(a, op, b);
            minOpNode->info  = presetValue;

            minOpNode->right->pop();
            minOpNode->right->pop();
          }
          else if(minOpType & ternaryOperatorType){
            minOpNode = minOpNode->left;

            std::string a  = minOpNode->value;
            std::string op = minOpNode->right->value;
            std::string b  = minOpNode->right->right->value;
            std::string c  = minOpNode->right->right->right->right->value;

            minOpNode->value = applyOperator(a, op, b, c);
            minOpNode->info  = presetValue;

            minOpNode->right->pop();
            minOpNode->right->pop();
            minOpNode->right->pop();
            minOpNode->right->pop();
          }
        }

        if(nodeRoot->right == NULL)
          return typeHolder(*nodeRoot);

        labelNodePos = nodeRoot;
      }

      // Shouldn't get here
      typeHolder th(nodeRoot->value);

      return th;
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
