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

namespace occa {
  namespace cli {
    namespace pretty {
      template <class TM>
      void printEntries(const std::string &title,
                        const std::vector<TM> &entries,
                        std::ostream &out) {
        if (!entries.size()) {
          return;
        }

        out << title << ":\n";

        int nameColumnWidth = 0;
        const int entryCount = (int) entries.size();

        // Get maximum size needed to print name
        for (int i = 0; i < entryCount; ++i) {
          const TM &entry = entries[i];
          const int nameSize = (int) entry.getPrintName().size();
          if (nameSize > nameColumnWidth) {
            nameColumnWidth = nameSize;
          }
        }
        nameColumnWidth += COLUMN_SPACING;
        if (nameColumnWidth > MAX_NAME_COLUMN_WIDTH) {
          nameColumnWidth = MAX_NAME_COLUMN_WIDTH;
        }

        std::stringstream ss;
        for (int i = 0; i < entryCount; ++i) {
          const TM &entry = entries[i];
          ss << "  " << entry.getPrintName();

          // If the line is larger than 'nameColumnWidth', start the
          //   description in the next line
          if ((int) ss.str().size() > (nameColumnWidth + COLUMN_SPACING)) {
            ss << '\n' << std::string(nameColumnWidth + COLUMN_SPACING, ' ');
          } else {
            ss << std::string(nameColumnWidth + COLUMN_SPACING - ss.str().size(), ' ');
          }

          out << ss.str();
          ss.str("");

          printDescription(out,
                           nameColumnWidth + COLUMN_SPACING,
                           MAX_DESC_COLUMN_WIDTH,
                           entry.description);
        }
        out << '\n';
      }
    }
  }
}
