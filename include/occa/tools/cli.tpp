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
