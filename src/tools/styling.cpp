/* The MIT License (MIT)
 *
 * Copyright (c) 2014-2017 David Medina and Tim Warburton
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

#include "occa/tools/styling.hpp"

namespace occa {
  namespace styling {
    std::string left(const std::string &str, const int width, const bool pad) {
      const int chars = (int) str.size();
      if (chars == 0 || width == 0) {
        return "";
      }
      const int spaces = (chars >= width) ? 0 : (width - chars);
      return (pad ? " " : "") + str + std::string(spaces + pad, ' ');
    }

    std::string right(const std::string &str, const int width, const bool pad) {
      const int chars = (int) str.size();
      if (chars == 0 || width == 0) {
        return "";
      }
      const int spaces = (chars >= width) ? 0 : (width - chars);
      return std::string(spaces + pad, ' ') + str + (pad ? " " : "");
    }

    std::string center(const std::string &str, const int width, const bool pad) {
      const int chars = (int) str.size();
      if (chars == 0 || width == 0) {
        return "";
      }
      const int spaces = (pad ? 2 : 0) + ((chars >= width) ? 0 : (width - chars));
      std::string ret = std::string((spaces + 0)/2, ' ');
      ret += str;
      ret += std::string((spaces + 1)/2, ' ');
      return ret;
    }

    //---[ field ]----------------------
    field::field(const std::string &name_, const std::string &value_) :
      name(name_),
      value(value_) {}
    //==================================


    //---[ fieldGroup ]-----------------
    fieldGroup::fieldGroup() {}

    udim_t fieldGroup::size() const {
      return fields.size();
    }

    fieldGroup& fieldGroup::add(const std::string &field,
                                const std::string &value) {
      fields.push_back(styling::field(field, value));
      return *this;
    }

    int fieldGroup::getFieldWidth() const {
      const int fieldCount = (int) fields.size();
      int maxWidth = 0;
      for (int i = 0; i < fieldCount; ++i) {
        const int iWidth = (int) fields[i].name.size();
        maxWidth = (maxWidth < iWidth) ? iWidth : maxWidth;
      }
      return maxWidth;
    }

    int fieldGroup::getValueWidth() const {
      const int fieldCount = (int) fields.size();
      int maxWidth = 0;
      for (int i = 0; i < fieldCount; ++i) {
        const int iWidth = (int) fields[i].value.size();
        maxWidth = (maxWidth < iWidth) ? iWidth : maxWidth;
      }
      return maxWidth;
    }
    //==================================


    //---[ section ]--------------------
    section::section(const std::string &name_) :
      name(name_) {
      groups.push_back(fieldGroup());
    }

    udim_t section::size() const {
      int fields = 0;
      std::vector<fieldGroup>::const_iterator it = groups.begin();
      while (it != groups.end()) {
        fields += it->size();
        ++it;
      }
      return fields;
    }

    section& section::add(const std::string &field, const std::string &value) {
      fieldGroup &fg = groups[groups.size() - 1];
      fg.add(field, value);
      return *this;
    }

    section& section::addDivider() {
      groups.push_back(fieldGroup());
      return *this;
    }

    int section::getFieldWidth() const {
      const int groupCount = (int) groups.size();
      int fieldWidth = 0;
      for (int i = 0; i < groupCount; ++i) {
        fieldWidth = std::max(fieldWidth, groups[i].getFieldWidth());
      }
      return fieldWidth;
    }

    int section::getValueWidth() const {
      const int groupCount = (int) groups.size();
      int valueWidth = 0;
      for (int i = 0; i < groupCount; ++i) {
        valueWidth = std::max(valueWidth, groups[i].getValueWidth());
      }
      return valueWidth;
    }

    std::string section::toString(const int sectionWidth,
                                  const int fieldWidth,
                                  const int valueWidth,
                                  const bool isFirstSection) const {
      std::stringstream ss;
      ss << std::string(sectionWidth + 2, '=') << 'o'
         << std::string(fieldWidth   + 2, '=') << 'o'
         << std::string(valueWidth   + 2, '=') << '\n';
      const std::string sectionDivider = ss.str();
      ss.str("");

      ss << std::string(sectionWidth + 2, ' ') << '|'
         << std::string(fieldWidth   + 2, '-') << '|'
         << std::string(valueWidth   + 2, '-') << '\n';
      const std::string groupDivider = ss.str();
      ss.str("");

      const int groupCount = (int) groups.size();
      if (isFirstSection) {
        ss << sectionDivider;
      }
      for (int i = 0; i < groupCount; ++i) {
        const fieldGroup &iGroup = groups[i];

        const int fieldCount = (int) iGroup.size();
        if (fieldCount == 0) {
          continue;
        }

        for (int j = 0; j < fieldCount; ++j) {
          const field& jField = iGroup.fields[j];
          if (i == 0 && j == 0) {
            ss << left(name, sectionWidth, true);
          } else {
            ss << std::string(sectionWidth + 2, ' ');
          }
          ss << '|'
             << left(jField.name, fieldWidth, true)
             << '|'
             << left(jField.value, valueWidth, true)
             << '\n';
        }
        if (i < (groupCount - 1)) {
          ss << groupDivider;
        }
      }
      ss << sectionDivider;

      return ss.str();
    }
    //==================================


    //---[ table ]----------------------
    table::table() {}

    void table::add(section &section) {
      sections.push_back(section);
    }

    std::string table::toString() const {
      const int sectionCount = (int) sections.size();
      std::string str;

      int sectionWidth = 0, fieldWidth = 0, valueWidth = 0;
      for (int i = 0; i < sectionCount; ++i) {
        const section &iSection = sections[i];
        sectionWidth = std::max(sectionWidth, (int) iSection.name.size());
        fieldWidth   = std::max(fieldWidth  , iSection.getFieldWidth());
        valueWidth   = std::max(valueWidth  , iSection.getValueWidth());
      }

      for (int i = 0; i < sectionCount; ++i) {
        if (sections[i].size()) {
          str += sections[i].toString(sectionWidth, fieldWidth, valueWidth, i == 0);
        }
      }

      return str;
    }

    std::ostream& operator << (std::ostream &out, const table &st) {
      out << st.toString();
      return out;
    }
    //==================================
  }
}
