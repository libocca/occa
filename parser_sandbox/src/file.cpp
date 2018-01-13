#include "file.hpp"
#include "tokenizer.hpp"
#include "occa/tools/io.hpp"

namespace occa {
  namespace lang {
    file_t::file_t(const std::string &filename_) :
      dirname(io::dirname(filename_)),
      filename(io::basename(filename_)),
      content(io::read(filename_)) {}

    filePosition::filePosition() :
      line(0),
      lineStart(NULL),
      pos(NULL) {}

    filePosition::filePosition(const int line_,
                               const char *lineStart_,
                               const char *pos_) :
      line(line_),
      lineStart(lineStart_),
      pos(pos_) {}

    filePosition::filePosition(const filePosition &other) :
      line(other.line),
      lineStart(other.lineStart),
      pos(other.pos) {}

    fileOrigin::fileOrigin() :
      fromInclude(true),
      file(NULL),
      position(),
      up(NULL) {}

    fileOrigin::fileOrigin(file_t *file_,
                           filePosition &position_) :
      fromInclude(true),
      file(file_),
      position(position_),
      up(NULL) {
      if (file) {
        file->addRef();
      }
    }

    fileOrigin::fileOrigin(fileOrigin &other) :
      fromInclude(other.fromInclude),
      file(other.file),
      position(other.position),
      up(other.up) {
      if (file) {
        file->addRef();
      }
      if (up) {
        up->addRef();
      }
    }

    fileOrigin::~fileOrigin() {
      if (file && !file->removeRef()) {
        delete file;
      }
      if (up && !up->removeRef()) {
        delete up;
      }
    }

    fileOrigin& fileOrigin::push(const bool fromInclude_,
                                 file_t *file_,
                                 filePosition &position_) {
      // TODO: fix
      return *this;
    }

    void fileOrigin::print(printer &pout) {
      // Print file location
      if (file) {
        pout << file->filename;
      } else {
        pout << "(source)";
      }
      pout << ':' << position.line;
      if (fromInclude) {
        pout << ':' << (position.pos - position.lineStart + 1);
      }
      if (!up) {
        return;
      }
      // Print connection from *up
      if (fromInclude) {
        pout << ": Included file:\n";
      } else {
        charStream stream(position.pos);
        // TODO: Add identifier
        // stream.skipIdentifier();
        pout << ": Expanded from macro '" << stream.str() << "':\n";
      }
    }
  }
}
