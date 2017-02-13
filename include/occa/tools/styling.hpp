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

#ifndef OCCA_TOOLS_STYLING_HEADER
#define OCCA_TOOLS_STYLING_HEADER

#include "occa/defines.hpp"
#include "occa/types.hpp"

#include <iostream>
#include <sstream>
#include <vector>

namespace occa {
  namespace styling {
    std::string left(const std::string &str, const int width, const bool pad = false);
    std::string right(const std::string &str, const int width, const bool pad = false);
    std::string center(const std::string &str, const int width, const bool pad = false);

    class field {
    public:
      std::string name, value;

      field(const std::string &name_, const std::string &value_ = "");
    };

    class fieldGroup {
    public:
      std::vector<field> fields;

      fieldGroup();

      udim_t size() const;
      fieldGroup& add(const std::string &field, const std::string &value = "");

      int getFieldWidth() const;
      int getValueWidth() const;
    };

    class section {
    public:
      std::string name;
      std::vector<fieldGroup> groups;

      section(const std::string &name_ = "");

      udim_t size() const;
      section& add(const std::string &field, const std::string &value = "");
      section& addDivider();

      int getFieldWidth() const;
      int getValueWidth() const;

      std::string toString(const int sectionWidth,
                           const int fieldWidth,
                           const int valueWidth,
                           const bool isFirstSection) const;
    };

    class table {
    public:
      std::vector<section> sections;

      table();

      void add(section &section);
      std::string toString() const;
      friend std::ostream& operator << (std::ostream &out, const table &ppt);
    };
    std::ostream& operator << (std::ostream &out, const table &ppt);
  }
}

#endif
