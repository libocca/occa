#ifndef OCCA_LANG_TOKEN_HEADER
#define OCCA_LANG_TOKEN_HEADER

#include <occa/io.hpp>
#include <occa/lang/file.hpp>
#include <occa/lang/type.hpp>

namespace occa {
  namespace lang {
    class operator_t;
    class token_t;
    class qualifier_t;
    class variable_t;

    typedef std::vector<token_t*> tokenVector;

    namespace charcodes {
      extern const char whitespace[];
      extern const char whitespaceNoNewline[];
      extern const char alpha[];
      extern const char number[];
      extern const char alphanumber[];
      extern const char identifierStart[];
      extern const char identifier[];
    }

    namespace encodingType {
      extern const int none;
      extern const int R;
      extern const int u8;
      extern const int u;
      extern const int U;
      extern const int L;
      extern const int ux;
      extern const int bits;
    }

    namespace tokenType {
      extern const int none;
      extern const int unknown;

      extern const int newline;

      extern const int systemHeader;
      extern const int header;

      extern const int directive;
      extern const int pragma;
      extern const int comment;

      extern const int identifier;

      extern const int qualifier;
      extern const int type;
      extern const int vartype;
      extern const int variable;
      extern const int function;

      extern const int primitive;
      extern const int op;

      extern const int char_;
      extern const int string;
      extern const int withUDF;
      extern const int withEncoding;
      extern const int encodingShift;

      int getEncoding(const int type);
      int mergeEncodings(const int type1, const int type2);
    }

    class token_t {
    public:
      fileOrigin origin;

      token_t(const fileOrigin &origin_);

      virtual ~token_t();

      template <class TM>
      inline bool is() const {
        return (dynamic_cast<const TM*>(this) != NULL);
      }

      template <class TM>
      inline TM& to() {
        TM *ptr = dynamic_cast<TM*>(this);
        OCCA_ERROR("Unable to cast token_t::to",
                   ptr != NULL);
        return *ptr;
      }

      template <class TM>
      inline const TM& to() const {
        const TM *ptr = dynamic_cast<const TM*>(this);
        OCCA_ERROR("Unable to cast token_t::to",
                   ptr != NULL);
        return *ptr;
      }

      static int safeType(token_t *token);
      static opType_t safeOperatorType(token_t *token);

      virtual int type() const = 0;

      virtual token_t* clone() const = 0;
      static token_t* clone(const token_t *token);

      opType_t getOpType();

      virtual void print(io::output &out) const = 0;

      void printWarning(const std::string &message) const;
      void printError(const std::string &message) const;

      std::string str() const;
      void debugPrint() const;
    };

    io::output& operator << (io::output &out,
                             token_t &token);

    //---[ Unknown ]--------------------
    class unknownToken : public token_t {
    public:
      char symbol;

      unknownToken(const fileOrigin &origin_);

      virtual ~unknownToken();

      virtual int type() const;

      virtual token_t* clone() const;

      virtual void print(io::output &out) const;
    };
    //==================================

    //---[ Newline ]--------------------
    class newlineToken : public token_t {
    public:
      newlineToken(const fileOrigin &origin_);

      virtual ~newlineToken();

      virtual int type() const;

      virtual token_t* clone() const;

      virtual void print(io::output &out) const;
    };
    //==================================

    //---[ Directive ]------------------
    class directiveToken : public token_t {
    public:
      std::string value;

      directiveToken(const fileOrigin &origin_,
                     const std::string &value_);

      virtual ~directiveToken();

      virtual int type() const;

      virtual token_t* clone() const;

      virtual void print(io::output &out) const;
    };
    //==================================

    //---[ Pragma ]---------------------
    class pragmaToken : public token_t {
    public:
      std::string value;

      pragmaToken(const fileOrigin &origin_,
                  const std::string &value_);

      virtual ~pragmaToken();

      virtual int type() const;

      virtual token_t* clone() const;

      virtual void print(io::output &out) const;
    };
    //==================================

    //---[ Comment ]-----------------
    class commentToken : public token_t {
    public:
      std::string value;

      commentToken(const fileOrigin &origin_,
                   const std::string &value_);

      virtual ~commentToken();

      virtual int type() const;

      virtual token_t* clone() const;

      virtual void print(io::output &out) const;
    };
    //==================================

    //---[ Identifier ]-----------------
    class identifierToken : public token_t {
    public:
      std::string value;

      identifierToken(const fileOrigin &origin_,
                      const std::string &value_);

      virtual ~identifierToken();

      virtual int type() const;

      virtual token_t* clone() const;

      virtual void print(io::output &out) const;
    };
    //==================================

    //---[ Qualifier ]------------------
    class qualifierToken : public token_t {
    public:
      const qualifier_t &qualifier;

      qualifierToken(const fileOrigin &origin_,
                     const qualifier_t &qualifier_);

      virtual ~qualifierToken();

      virtual int type() const;

      virtual token_t* clone() const;

      virtual void print(io::output &out) const;
    };
    //==================================

    //---[ Type ]-----------------------
    class typeToken : public token_t {
    public:
      type_t &value;

      typeToken(const fileOrigin &origin_,
                type_t &type_);

      virtual ~typeToken();

      virtual int type() const;

      virtual token_t* clone() const;

      virtual void print(io::output &out) const;
    };
    //==================================

    //---[ Vartype ]--------------------
    class vartypeToken : public token_t {
    public:
      vartype_t value;

      vartypeToken(const fileOrigin &origin_,
                   const vartype_t &value_);

      virtual ~vartypeToken();

      virtual int type() const;

      virtual token_t* clone() const;

      virtual void print(io::output &out) const;
    };
    //==================================

    //---[ Variable ]-------------------
    class variableToken : public token_t {
    public:
      variable_t &value;

      variableToken(const fileOrigin &origin_,
                    variable_t &variable);

      virtual ~variableToken();

      virtual int type() const;

      virtual token_t* clone() const;

      virtual void print(io::output &out) const;
    };
    //==================================

    //---[ Function ]-------------------
    class functionToken : public token_t {
    public:
      function_t &value;

      functionToken(const fileOrigin &origin_,
                    function_t &function);

      virtual ~functionToken();

      virtual int type() const;

      virtual token_t* clone() const;

      virtual void print(io::output &out) const;
    };
    //==================================

    //---[ Primitive ]------------------
    class primitiveToken : public token_t {
    public:
      primitive value;
      std::string strValue;

      primitiveToken(const fileOrigin &origin_,
                     const primitive &value_,
                     const std::string &strValue_);

      virtual ~primitiveToken();

      virtual int type() const;

      virtual token_t* clone() const;

      virtual void print(io::output &out) const;
    };
    //==================================

    //---[ Operator ]-------------------
    class operatorToken : public token_t {
    public:
      const operator_t *op;

      operatorToken(const fileOrigin &origin_,
                    const operator_t &op_);

      virtual ~operatorToken();

      virtual int type() const;

      virtual const opType_t& opType() const;

      virtual token_t* clone() const;

      virtual void print(io::output &out) const;
    };
    //==================================

    //---[ Char ]-----------------------
    class charToken : public token_t {
    public:
      int encoding;
      std::string value;
      std::string udf;

      charToken(const fileOrigin &origin_,
                int encoding_,
                const std::string &value_,
                const std::string &udf_);

      virtual ~charToken();

      virtual int type() const;

      virtual token_t* clone() const;

      virtual void print(io::output &out) const;
    };
    //==================================

    //---[ String ]---------------------
    class stringToken : public token_t {
    public:
      int encoding;
      std::string value;
      std::string udf;

      stringToken(const fileOrigin &origin_,
                  const std::string &value_);

      stringToken(const fileOrigin &origin_,
                  int encoding_,
                  const std::string &value_,
                  const std::string &udf_);

      virtual ~stringToken();

      virtual int type() const;

      virtual token_t* clone() const;

      void append(const stringToken &token);

      virtual void print(io::output &out) const;
    };
    //==================================

    //---[ Helper Methods ]-------------
    void freeTokenVector(tokenVector &tokens);

    std::string stringifyTokens(tokenVector &tokens,
                                const bool addSpaces);
    //==================================
  }
}

#endif
