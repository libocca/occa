#ifndef OCCA_PARSER_FILE_HEADER2
#define OCCA_PARSER_FILE_HEADER2

#include <iostream>
#include "occa/tools/gc.hpp"
#include "operator.hpp"

namespace occa {
  namespace lang {
    class file_t : public withRefs {
    public:
      std::string dirname;
      std::string filename;
      std::string content;

      file_t(const std::string &filename_);
    };

    class filePosition : public withRefs {
    public:
      int line;
      const char *lineStart;
      const char *pos;

      filePosition();

      filePosition(const int line_,
                   const char *lineStart_,
                   const char *pos_);

      filePosition(const filePosition &other);
    };

    class fileOrigin {
    public:
      bool fromInclude;
      file_t *file;
      filePosition position;

      filePosition *up;

      fileOrigin();

      fileOrigin(file_t *file_,
                 filePosition &position_);

      fileOrigin(fileOrigin &other);

      ~fileOrigin();

      fileOrigin& push(const bool fromInclude_,
                       file_t *file_,
                       filePosition &position_);

      void print(printer &pout);
    };
  }
}

#endif
