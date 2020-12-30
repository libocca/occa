#ifndef OCCA_INTERNAL_LANG_TOKEN_TOKEN_HEADER
#define OCCA_INTERNAL_LANG_TOKEN_TOKEN_HEADER

#include <occa/internal/io.hpp>
#include <occa/internal/lang/file.hpp>
#include <occa/internal/lang/type.hpp>

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

    //---[ Helper Methods ]-------------
    void freeTokenVector(tokenVector &tokens);

    std::string stringifyTokens(tokenVector &tokens,
                                const bool addSpaces);
    //==================================
  }
}

#endif
