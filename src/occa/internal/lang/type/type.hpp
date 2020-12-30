#ifndef OCCA_INTERNAL_LANG_TYPE_TYPE_HEADER
#define OCCA_INTERNAL_LANG_TYPE_TYPE_HEADER

#include <vector>

#include <occa/dtype.hpp>
#include <occa/internal/io/output.hpp>
#include <occa/internal/lang/attribute.hpp>
#include <occa/internal/lang/printer.hpp>
#include <occa/internal/lang/qualifier.hpp>

namespace occa {
  namespace lang {
    class token_t;
    class identifierToken;
    class operatorToken;
    class exprNode;
    class blockStatement;
    class pointer_t;
    class array_t;
    class variable_t;

    typedef std::vector<array_t>     arrayVector;
    typedef std::vector<pointer_t>   pointerVector;
    typedef std::vector<variable_t>  variableVector;
    typedef std::vector<variable_t*> variablePtrVector;

    namespace typeType {
      extern const int none;

      extern const int primitive;
      extern const int typedef_;
      extern const int functionPtr;
      extern const int function;

      extern const int class_;
      extern const int struct_;
      extern const int union_;
      extern const int enum_;
      extern const int structure;
    }

    namespace classAccess {
      extern const int private_;
      extern const int protected_;
      extern const int public_;
    }

    class type_t {
    public:
      identifierToken *source;
      attributeTokenMap attributes;

      type_t(const std::string &name_ = "");
      type_t(const identifierToken &source_);
      type_t(const type_t &other);
      virtual ~type_t();

      virtual int type() const = 0;
      virtual type_t& clone() const = 0;

      template <class TM>
      inline bool is() const {
        return (dynamic_cast<const TM*>(this) != NULL);
      }

      template <class TM>
      inline TM& to() {
        TM *ptr = dynamic_cast<TM*>(this);
        OCCA_ERROR("Unable to cast type_t::to",
                   ptr != NULL);
        return *ptr;
      }

      template <class TM>
      inline const TM& to() const {
        const TM *ptr = dynamic_cast<const TM*>(this);
        OCCA_ERROR("Unable to cast type_t::to",
                   ptr != NULL);
        return *ptr;
      }

      template <class TM>
      inline TM& constTo() const {
        TM *ptr = dynamic_cast<TM*>(const_cast<type_t*>(this));
        OCCA_ERROR("Unable to cast type_t::to",
                   ptr != NULL);
        return *ptr;
      }

      void setSource(const identifierToken &source_);

      virtual const std::string& name() const;
      virtual bool isNamed() const;

      virtual bool isPointerType() const;

      virtual dtype_t dtype() const = 0;

      bool operator == (const type_t &other) const;
      bool operator != (const type_t &other) const;
      virtual bool equals(const type_t &other) const;

      bool hasAttribute(const std::string &attr) const;

      virtual void printDeclaration(printer &pout) const = 0;

      void printWarning(const std::string &message) const;
      void printError(const std::string &message) const;
    };

    io::output& operator << (io::output &out,
                             const type_t &type);

    printer& operator << (printer &pout,
                          const type_t &type);
  }
}

#endif
