#include <occa/internal/utils/styling.hpp>

namespace occa {
  namespace styling {
    std::string left(const std::string &str,
                     const int width,
                     const bool pad) {
      const int chars = (int) str.size();
      if (chars == 0 || width == 0) {
        return "";
      }
      const int spaces = (chars >= width) ? 0 : (width - chars);
      return (pad ? " " : "") + str + std::string(spaces + pad, ' ');
    }

    std::string right(const std::string &str,
                      const int width,
                      const bool pad) {
      const int chars = (int) str.size();
      if (chars == 0 || width == 0) {
        return "";
      }
      const int spaces = (chars >= width) ? 0 : (width - chars);
      return std::string(spaces + pad, ' ') + str + (pad ? " " : "");
    }

    //---[ field ]----------------------
    field::field(const std::string &name_,
                 const std::string &value_) :
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
      int maxWidth = 0;
      for (auto &field: fields) {
        maxWidth = std::max(maxWidth, (int) field.name.size());
      }
      return maxWidth;
    }

    int fieldGroup::getValueWidth() const {
      int maxWidth = 0;
      for (auto &field: fields) {
        maxWidth = std::max(maxWidth, (int) field.value.size());
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

    section& section::add(const std::string &field,
                          const std::string &value) {
      groups.back().add(field, value);
      return *this;
    }

    section& section::addDivider() {
      groups.push_back(fieldGroup());
      return *this;
    }

    int section::getFieldWidth() const {
      int fieldWidth = 0;
      for (auto &group : groups) {
        fieldWidth = std::max(fieldWidth, group.getFieldWidth());
      }
      return fieldWidth;
    }

    int section::getValueWidth() const {
      int valueWidth = 0;
      for (auto &group : groups) {
        valueWidth = std::max(valueWidth, group.getValueWidth());
      }
      return valueWidth;
    }

    std::string section::toString(const int indent,
                                  const int sectionWidth,
                                  const int fieldWidth,
                                  const int valueWidth,
                                  const bool isFirstSection) const {
      const std::string indentStr(indent, ' ');

      std::stringstream ss;
      ss << indentStr
         << std::string(sectionWidth + 2, '=') << '+'
         << std::string(fieldWidth   + 2, '=') << '+'
         << std::string(valueWidth   + 2, '=') << '\n';
      const std::string sectionDivider = ss.str();
      ss.str("");

      ss << indentStr
         << std::string(sectionWidth + 2, ' ') << '|'
         << std::string(fieldWidth   + 2, '-') << '+'
         << std::string(valueWidth   + 2, '-') << '\n';
      const std::string groupDivider = ss.str();
      ss.str("");

      if (isFirstSection) {
        ss << sectionDivider;
      }

      const int groupCount = (int) groups.size();
      for (int i = 0; i < groupCount; ++i) {
        const fieldGroup &group = groups[i];

        const int fieldCount = (int) group.size();
        if (fieldCount == 0) {
          continue;
        }

        for (int j = 0; j < fieldCount; ++j) {
          const field& field = group.fields[j];

          ss << indentStr;
          if (i == 0 && j == 0) {
            ss << left(name, sectionWidth, true);
          } else {
            ss << std::string(sectionWidth + 2, ' ');
          }

          ss << '|'
             << left(field.name, fieldWidth, true)
             << '|'
             << left(field.value, valueWidth, true)
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

    std::string table::toString(const int indent) const {
      int sectionWidth = 0, fieldWidth = 0, valueWidth = 0;
      for (auto &section : sections) {
        sectionWidth = std::max(sectionWidth, (int) section.name.size());
        fieldWidth   = std::max(fieldWidth  , section.getFieldWidth());
        valueWidth   = std::max(valueWidth  , section.getValueWidth());
      }

      std::string str;
      bool isFirstSection = true;
      for (auto &section : sections) {
        if (section.size()) {
          str += section.toString(
            indent,
            sectionWidth,
            fieldWidth,
            valueWidth,
            isFirstSection
          );
        }
        isFirstSection = false;
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
