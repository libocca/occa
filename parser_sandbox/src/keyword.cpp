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
#include "keyword.hpp"
#include "variable.hpp"
#include "builtins/types.hpp"

namespace occa {
  namespace lang {
    namespace keywordType {
      const int none        = (1 << 0);

      const int qualifier   = (1 << 1);
      const int type        = (1 << 2);
      const int variable    = (1 << 3);

      const int if_         = (1 << 4);
      const int else_       = (1 << 5);
      const int switch_     = (1 << 6);
      const int conditional = (if_   |
                               else_ |
                               switch_);

      const int case_       = (1 << 7);
      const int default_    = (1 << 8);
      const int switchLabel = (case_ |
                               default_);

      const int for_        = (1 << 9);
      const int while_      = (1 << 10);
      const int do_         = (1 << 11);
      const int iteration   = (for_   |
                               while_ |
                               do_);

      const int break_      = (1 << 12);
      const int continue_   = (1 << 13);
      const int return_     = (1 << 14);
      const int goto_       = (1 << 15);
      const int jump        = (break_    |
                               continue_ |
                               return_   |
                               goto_);

      const int namespace_  = (1 << 16);

      const int public_     = (1 << 17);
      const int protected_  = (1 << 18);
      const int private_    = (1 << 19);
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

    int keyword_t::safeType(keyword_t *keyword) {
      return (keyword
              ? keyword->type()
              : keywordType::none);
    }

    //---[ Qualifier ]------------------
    qualifierKeyword::qualifierKeyword(const qualifier_t &qualifier_) :
      qualifier(qualifier_) {}

    int qualifierKeyword::type() {
      return keywordType::qualifier;
    }

    std::string qualifierKeyword::name() {
      return qualifier.name;
    }
    //==================================

    //---[ Type ]-----------------------
    typeKeyword::typeKeyword(const type_t &type__) :
      type_(type__) {}

    int typeKeyword::type() {
      return keywordType::type;
    }

    std::string typeKeyword::name() {
      return type_.name();
    }
    //==================================

    //---[ Variable ]-------------------
    variableKeyword::variableKeyword(const variable &var_) :
      var(var_) {}

    int variableKeyword::type() {
      return keywordType::variable;
    }

    std::string variableKeyword::name() {
      return var.name();
    }
    //==================================

    //---[ Statement ]------------------
    statementKeyword::statementKeyword(const int sType_,
                                       const std::string &sName_) :
      sType(sType_),
      sName(sName_) {}

    int statementKeyword::type() {
      return sType;
    }

    std::string statementKeyword::name() {
      return sName;
    }
    //==================================

    void getKeywords(keywordTrie &keywords) {
      // Qualifiers
      addKeyword(keywords, new qualifierKeyword(const_));
      addKeyword(keywords, new qualifierKeyword(constexpr_));
      addKeyword(keywords, new qualifierKeyword(restrict_));
      addKeyword(keywords, new qualifierKeyword(friend_));
      addKeyword(keywords, new qualifierKeyword(typedef_));
      addKeyword(keywords, new qualifierKeyword(signed_));
      addKeyword(keywords, new qualifierKeyword(unsigned_));
      addKeyword(keywords, new qualifierKeyword(volatile_));
      addKeyword(keywords, new qualifierKeyword(long_));
      addKeyword(keywords, new qualifierKeyword(longlong_));

      addKeyword(keywords, new qualifierKeyword(extern_));
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
      addKeyword(keywords, new typeKeyword(bool_));
      addKeyword(keywords, new typeKeyword(char_));
      addKeyword(keywords, new typeKeyword(char16_t_));
      addKeyword(keywords, new typeKeyword(char32_t_));
      addKeyword(keywords, new typeKeyword(wchar_t_));
      addKeyword(keywords, new typeKeyword(short_));
      addKeyword(keywords, new typeKeyword(int_));
      addKeyword(keywords, new typeKeyword(float_));
      addKeyword(keywords, new typeKeyword(double_));
      addKeyword(keywords, new typeKeyword(void_));
      addKeyword(keywords, new typeKeyword(auto_));

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

    void freeKeywords(keywordTrie &keywords) {
      keywords.freeze();
      const int count = keywords.size();
      for (int i = 0; i < count; ++i) {
        delete keywords.values[i];
      }
      keywords.clear();
    }
  }
}
