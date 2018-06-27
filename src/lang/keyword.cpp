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
#include <occa/lang/keyword.hpp>
#include <occa/lang/variable.hpp>
#include <occa/lang/builtins/types.hpp>

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
      occa::printError(std::cerr, message);
    }

    int keyword_t::safeType(keyword_t *keyword) {
      return (keyword
              ? keyword->type()
              : keywordType::none);
    }

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
      if (type_.type() & typeType::typedef_) {
        delete &type_;
      }
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

    void getKeywords(keywordMap &keywords) {
      // Qualifiers
      addKeyword(keywords, new qualifierKeyword(const_));
      addKeyword(keywords, new qualifierKeyword(constexpr_));
      addKeyword(keywords, new qualifierKeyword(friend_));
      addKeyword(keywords, new qualifierKeyword(typedef_));
      addKeyword(keywords, new qualifierKeyword(signed_));
      addKeyword(keywords, new qualifierKeyword(unsigned_));
      addKeyword(keywords, new qualifierKeyword(volatile_));
      addKeyword(keywords, new qualifierKeyword(long_));
      addKeyword(keywords, new qualifierKeyword(longlong_));

      addKeyword(keywords, new qualifierKeyword(extern_));
      addKeyword(keywords, new qualifierKeyword(externC));
      addKeyword(keywords, new qualifierKeyword(externCpp));
      addKeyword(keywords, new qualifierKeyword(mutable_));
      addKeyword(keywords, new qualifierKeyword(register_));
      addKeyword(keywords, new qualifierKeyword(static_));
      addKeyword(keywords, new qualifierKeyword(thread_local_));

      addKeyword(keywords, new qualifierKeyword(explicit_));
      addKeyword(keywords, new qualifierKeyword(inline_));
      addKeyword(keywords, new qualifierKeyword(virtual_));

      addKeyword(keywords, new qualifierKeyword(class_));
      addKeyword(keywords, new qualifierKeyword(struct_));
      addKeyword(keywords, new qualifierKeyword(enum_));
      addKeyword(keywords, new qualifierKeyword(union_));

      // Types
      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(bool_)));
      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(char_)));
      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(char16_t_)));
      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(char32_t_)));
      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(wchar_t_)));
      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(short_)));
      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(int_)));
      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(float_)));
      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(double_)));
      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(void_)));
      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(auto_)));

      // OKL Types
      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(uchar2)));
      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(uchar3)));
      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(uchar4)));

      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(char2)));
      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(char3)));
      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(char4)));

      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(ushort2)));
      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(ushort3)));
      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(ushort4)));

      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(short2)));
      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(short3)));
      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(short4)));

      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(uint2)));
      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(uint3)));
      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(uint4)));

      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(int2)));
      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(int3)));
      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(int4)));

      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(ulong2)));
      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(ulong3)));
      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(ulong4)));

      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(long2)));
      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(long3)));
      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(long4)));

      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(float2)));
      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(float3)));
      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(float4)));

      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(double2)));
      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(double3)));
      addKeyword(keywords, new typeKeyword(const_cast<primitive_t&>(double4)));

      // TODO: Add builtin functions
      //  - const_cast, static_cast, dynamic_cast, reinterpret_cast
      //  - typeid
      //  - noexcept
      //  - alignof
      //  - sizeof...

      // Conditional statements
      addKeyword(keywords, new statementKeyword(keywordType::if_       , "if"));
      addKeyword(keywords, new statementKeyword(keywordType::else_     , "else"));
      addKeyword(keywords, new statementKeyword(keywordType::switch_   , "switch"));
      addKeyword(keywords, new statementKeyword(keywordType::case_     , "case"));
      addKeyword(keywords, new statementKeyword(keywordType::default_  , "default"));
      // Iteration statements
      addKeyword(keywords, new statementKeyword(keywordType::for_      , "for"));
      addKeyword(keywords, new statementKeyword(keywordType::while_    , "while"));
      addKeyword(keywords, new statementKeyword(keywordType::do_       , "do"));
      // Jump statements
      addKeyword(keywords, new statementKeyword(keywordType::break_    , "break"));
      addKeyword(keywords, new statementKeyword(keywordType::continue_ , "continue"));
      addKeyword(keywords, new statementKeyword(keywordType::return_   , "return"));
      addKeyword(keywords, new statementKeyword(keywordType::goto_     , "goto"));
      // Misc
      addKeyword(keywords, new statementKeyword(keywordType::namespace_, "namespace"));
      // Class access statements
      addKeyword(keywords, new statementKeyword(keywordType::public_   , "public"));
      addKeyword(keywords, new statementKeyword(keywordType::protected_, "protected"));
      addKeyword(keywords, new statementKeyword(keywordType::private_  , "private"));
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

    void replaceKeyword(keywordMap &keywords,
                        keyword_t *keyword,
                        const bool deleteSource) {
      if (!keyword) {
        return;
      }
      // Ignore keywords without names
      const std::string &name = keyword->name();
      if (!name.size()) {
        delete keyword;
        return;
      }
      keywordMap::iterator it = keywords.find(name);
      if (it != keywords.end()) {
        // Make sure we aren't overriding ourselves
        if (it->second == keyword) {
          return;
        }
        // Free last keyword
        if (deleteSource) {
          it->second->deleteSource();
        }
        delete it->second;
      }
      keywords[name] = keyword;
    }
  }
}
