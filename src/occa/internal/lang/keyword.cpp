#include <occa/internal/lang/keyword.hpp>
#include <occa/internal/lang/statementContext.hpp>
#include <occa/internal/lang/token.hpp>
#include <occa/internal/lang/variable.hpp>
#include <occa/internal/lang/builtins/types.hpp>

namespace occa {
  namespace lang {
    namespace keywordType {
      const int none        = (1 << 0);

      const int qualifier   = (1 << 1);
      const int type        = (1 << 2);
      const int variable    = (1 << 3);
      const int function    = (1 << 4);

      const int if_         = (1 << 5);
      const int else_       = (1 << 6);
      const int switch_     = (1 << 7);
      const int conditional = (if_   |
                               else_ |
                               switch_);

      const int case_       = (1 << 8);
      const int default_    = (1 << 9);
      const int switchLabel = (case_ |
                               default_);

      const int for_        = (1 << 10);
      const int while_      = (1 << 11);
      const int do_         = (1 << 12);
      const int iteration   = (for_   |
                               while_ |
                               do_);

      const int break_      = (1 << 13);
      const int continue_   = (1 << 14);
      const int return_     = (1 << 15);
      const int goto_       = (1 << 16);
      const int jump        = (break_    |
                               continue_ |
                               return_   |
                               goto_);

      const int namespace_  = (1 << 17);

      const int public_     = (1 << 18);
      const int protected_  = (1 << 19);
      const int private_    = (1 << 20);
      const int classAccess = (public_    |
                               protected_ |
                               private_);

      const int statement   = (conditional |
                               switchLabel |
                               iteration   |
                               jump        |
                               namespace_  |
                               classAccess);
    }

    keyword_t::~keyword_t() {}

    int keyword_t::type() const {
      return keywordType::none;
    }

    const std::string& keyword_t::name() {
      static std::string empty;
      return empty;
    }

    keyword_t* keyword_t::clone() const {
      return new keyword_t();
    }

    void keyword_t::deleteSource() {}

    void keyword_t::printError(const std::string &message) {
      occa::printError(io::stderr, message);
    }

    int keyword_t::safeType(keyword_t *keyword) {
      return (keyword
              ? keyword->type()
              : keywordType::none);
    }

    //---[ Keywords ]-------------------
    keywords_t::keywords_t() {}

    void keywords_t::free(const bool deleteSource) {
      freeKeywords(keywords, deleteSource);
    }

    keywordMapIterator keywords_t::begin() {
      return keywords.begin();
    }

    keywordMapIterator keywords_t::end() {
      return keywords.end();
    }

    keyword_t& keywords_t::get(statementContext_t &smntContext,
                               token_t *token) const {
      static keyword_t noKeyword;
      if (!token) {
        return noKeyword;
      }

      const int tType = token->type();
      if (!(tType & (tokenType::identifier |
                     tokenType::qualifier  |
                     tokenType::type       |
                     tokenType::variable   |
                     tokenType::function))) {
        return noKeyword;
      }

      std::string name;
      if (tType & tokenType::identifier) {
        name = token->to<identifierToken>().value;
      }
      else if (tType & tokenType::qualifier) {
        name = token->to<qualifierToken>().qualifier.name;
      }
      else if (tType & tokenType::type) {
        name = token->to<typeToken>().value.name();
      }
      else if (tType & tokenType::variable) {
        name = token->to<variableToken>().value.name();
      }
      else if (tType & tokenType::function) {
        name = token->to<functionToken>().value.name();
      }

      return get(smntContext, name);
    }

    keyword_t& keywords_t::get(statementContext_t &smntContext,
                               const std::string &name) const {
      static keyword_t noKeyword;

      cKeywordMapIterator it = keywords.find(name);
      if (it != keywords.end()) {
        return *(it->second);
      }
      if (smntContext.up) {
        return smntContext.up->getScopeKeyword(name);
      }
      return noKeyword;
    }

    void freeKeywords(keywordMap &keywords,
                      const bool deleteSource) {
      keywordMapIterator it = keywords.begin();
      while (it != keywords.end()) {
        keyword_t &keyword = *(it->second);
        if (deleteSource) {
          keyword.deleteSource();
        }
        delete &keyword;
        ++it;
      }
      keywords.clear();
    }
    //==================================

    //---[ Qualifier ]------------------
    qualifierKeyword::qualifierKeyword(const qualifier_t &qualifier_) :
      qualifier(qualifier_) {}

    int qualifierKeyword::type() const {
      return keywordType::qualifier;
    }

    const std::string& qualifierKeyword::name() {
      return qualifier.name;
    }

    keyword_t* qualifierKeyword::clone() const {
      return new qualifierKeyword(qualifier);
    }
    //==================================

    //---[ Type ]-----------------------
    typeKeyword::typeKeyword(type_t &type__) :
      type_(type__) {}

    int typeKeyword::type() const {
      return keywordType::type;
    }

    const std::string& typeKeyword::name() {
      return type_.name();
    }

    keyword_t* typeKeyword::clone() const {
      return new typeKeyword(type_.clone());
    }

    void typeKeyword::deleteSource() {
      // TODO: Make sure the typedef type is being deleted
      // if (type_.type() & typeType::typedef_) {
      //   delete &type_;
      // }
    }

    void typeKeyword::printError(const std::string &message) {
      type_.printError(message);
    }
    //==================================

    //---[ Variable ]-------------------
    variableKeyword::variableKeyword(variable_t &variable_) :
      variable(variable_) {}

    int variableKeyword::type() const {
      return keywordType::variable;
    }

    const std::string& variableKeyword::name() {
      return variable.name();
    }

    keyword_t* variableKeyword::clone() const {
      return new variableKeyword(variable.clone());
    }

    void variableKeyword::deleteSource() {
      delete &variable;
    }

    void variableKeyword::printError(const std::string &message) {
      variable.printError(message);
    }
    //==================================

    //---[ Function ]-------------------
    functionKeyword::functionKeyword(function_t &function_) :
      function(function_) {}

    int functionKeyword::type() const {
      return keywordType::function;
    }

    const std::string& functionKeyword::name() {
      return function.name();
    }

    keyword_t* functionKeyword::clone() const {
      return new functionKeyword((function_t&) function.clone());
    }

    void functionKeyword::deleteSource() {
      delete &function;
    }

    void functionKeyword::printError(const std::string &message) {
      function.printError(message);
    }
    //==================================

    //---[ Statement ]------------------
    statementKeyword::statementKeyword(const int sType_,
                                       const std::string &sName_) :
      sType(sType_),
      sName(sName_) {}

    int statementKeyword::type() const {
      return sType;
    }

    const std::string& statementKeyword::name() {
      return sName;
    }

    keyword_t* statementKeyword::clone() const {
      return new statementKeyword(sType, sName);
    }
    //==================================

    void getKeywords(keywords_t &keywords) {
      // Qualifiers
      keywords.add(*(new qualifierKeyword(const_)));
      keywords.add(*(new qualifierKeyword(constexpr_)));
      keywords.add(*(new qualifierKeyword(friend_)));
      keywords.add(*(new qualifierKeyword(typedef_)));
      keywords.add(*(new qualifierKeyword(signed_)));
      keywords.add(*(new qualifierKeyword(unsigned_)));
      keywords.add(*(new qualifierKeyword(volatile_)));
      keywords.add(*(new qualifierKeyword(long_)));
      keywords.add(*(new qualifierKeyword(longlong_)));

      keywords.add(*(new qualifierKeyword(extern_)));
      keywords.add(*(new qualifierKeyword(externC)));
      keywords.add(*(new qualifierKeyword(externCpp)));
      keywords.add(*(new qualifierKeyword(mutable_)));
      keywords.add(*(new qualifierKeyword(register_)));
      keywords.add(*(new qualifierKeyword(static_)));
      keywords.add(*(new qualifierKeyword(thread_local_)));

      keywords.add(*(new qualifierKeyword(explicit_)));
      keywords.add(*(new qualifierKeyword(inline_)));
      keywords.add(*(new qualifierKeyword(virtual_)));

      keywords.add(*(new qualifierKeyword(class_)));
      keywords.add(*(new qualifierKeyword(struct_)));
      keywords.add(*(new qualifierKeyword(enum_)));
      keywords.add(*(new qualifierKeyword(union_)));

      // Types
      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(bool_))));
      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(char_))));
      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(char16_t_))));
      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(char32_t_))));
      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(wchar_t_))));
      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(short_))));
      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(int_))));
      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(float_))));
      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(double_))));
      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(void_))));
      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(auto_))));
      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(size_t_))));
      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(ptrdiff_t_))));

      // OKL Types
      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(uchar2))));
      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(uchar3))));
      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(uchar4))));

      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(char2))));
      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(char3))));
      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(char4))));

      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(ushort2))));
      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(ushort3))));
      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(ushort4))));

      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(short2))));
      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(short3))));
      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(short4))));

      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(uint2))));
      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(uint3))));
      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(uint4))));

      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(int2))));
      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(int3))));
      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(int4))));

      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(ulong2))));
      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(ulong3))));
      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(ulong4))));

      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(long2))));
      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(long3))));
      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(long4))));

      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(float2))));
      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(float3))));
      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(float4))));

      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(double2))));
      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(double3))));
      keywords.add(*(new typeKeyword(const_cast<primitive_t&>(double4))));

      // TODO: Add builtin functions
      //  - const_cast, static_cast, dynamic_cast, reinterpret_cast
      //  - typeid
      //  - noexcept
      //  - alignof
      //  - sizeof...

      // Conditional statements
      keywords.add(*(new statementKeyword(keywordType::if_       , "if")));
      keywords.add(*(new statementKeyword(keywordType::else_     , "else")));
      keywords.add(*(new statementKeyword(keywordType::switch_   , "switch")));
      keywords.add(*(new statementKeyword(keywordType::case_     , "case")));
      keywords.add(*(new statementKeyword(keywordType::default_  , "default")));
      // Iteration statements
      keywords.add(*(new statementKeyword(keywordType::for_      , "for")));
      keywords.add(*(new statementKeyword(keywordType::while_    , "while")));
      keywords.add(*(new statementKeyword(keywordType::do_       , "do")));
      // Jump statements
      keywords.add(*(new statementKeyword(keywordType::break_    , "break")));
      keywords.add(*(new statementKeyword(keywordType::continue_ , "continue")));
      keywords.add(*(new statementKeyword(keywordType::return_   , "return")));
      keywords.add(*(new statementKeyword(keywordType::goto_     , "goto")));
      // Misc
      keywords.add(*(new statementKeyword(keywordType::namespace_, "namespace")));
      // Class access statements
      keywords.add(*(new statementKeyword(keywordType::public_   , "public")));
      keywords.add(*(new statementKeyword(keywordType::protected_, "protected")));
      keywords.add(*(new statementKeyword(keywordType::private_  , "private")));
    }
  }
}
