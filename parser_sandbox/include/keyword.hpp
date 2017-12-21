#ifndef OCCA_PARSER_KEYWORD_HEADER2
#define OCCA_PARSER_KEYWORD_HEADER2

#include "occa/defines.hpp"
#include "occa/types.hpp"
#include "occa/tools/sys.hpp"

#include "type.hpp"

namespace occa {
  namespace lang {
    class keywordType {
    public:
      static const int none      = 0;
      static const int qualifier = (1 << 0);
      static const int primitive = (1 << 1);
      static const int typedef_  = (1 << 2);
      static const int class_    = (1 << 3);
      static const int function_ = (1 << 4);
      static const int attribute = (1 << 5);
    };

    class keyword_t {
    public:
      int ktype;
      specifier *ptr;

      keyword_t();
      keyword_t(const int ktype_, specifier *ptr_);

      inline bool isQualifier() {
        return (ktype == keywordType::qualifier);
      }

      inline bool isPrimitive() {
        return (ktype == keywordType::primitive);
      }

      inline bool isTypedef_() {
        return (ktype == keywordType::typedef_);
      }

      inline bool isClass_() {
        return (ktype == keywordType::class_);
      }

      inline bool isFunction_() {
        return (ktype == keywordType::function_);
      }

      inline bool isAttribute() {
        return (ktype == keywordType::attribute);
      }

      inline class qualifier& qualifier() {
        return *((class qualifier*) ptr);
      }

      inline primitiveType& primitive() {
        return *((primitiveType*) ptr);
      }

      inline typedefType& typedef_() {
        return *dynamic_cast<typedefType*>(ptr);
      }

      inline classType& class_() {
        return *dynamic_cast<classType*>(ptr);
      }

      inline functionType& function_() {
        return *dynamic_cast<functionType*>(ptr);
      }

      inline class attribute& attribute() {
        return *((class attribute*) ptr);
      }
    };
  }
}

#endif
