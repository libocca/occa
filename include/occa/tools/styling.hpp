#ifndef OCCA_INTERNAL_UTILS_STYLING_HEADER
#define OCCA_INTERNAL_UTILS_STYLING_HEADER

#include <occa/defines.hpp>
#include <occa/types.hpp>

#include <iostream>
#include <sstream>
#include <vector>

namespace occa {
  namespace styling {
    std::string left(const std::string &str,
                     const int width,
                     const bool pad = false);
    std::string right(const std::string &str,
                      const int width,
                      const bool pad = false);

    class field {
    public:
      std::string name, value;

      field(const std::string &name_,
            const std::string &value_ = "");
    };

    class fieldGroup {
    public:
      std::vector<field> fields;

      fieldGroup();

      udim_t size() const;
      fieldGroup& add(const std::string &field,
                      const std::string &value = "");

      int getFieldWidth() const;
      int getValueWidth() const;
    };

    class section {
    public:
      std::string name;
      std::vector<fieldGroup> groups;

      section(const std::string &name_ = "");

      udim_t size() const;
      section& add(const std::string &field,
                   const std::string &value = "");
      section& addDivider();

      int getFieldWidth() const;
      int getValueWidth() const;

      std::string toString(const int indent,
                           const int sectionWidth,
                           const int fieldWidth,
                           const int valueWidth,
                           const bool isFirstSection) const;
    };

    class table {
    public:
      std::vector<section> sections;

      table();

      void add(section &section);
      std::string toString(const int indent = 4) const;

      friend std::ostream& operator << (std::ostream &out,
                                      const table &ppt);
    };

    std::ostream& operator << (std::ostream &out,
                             const table &ppt);
  }
}

#endif
