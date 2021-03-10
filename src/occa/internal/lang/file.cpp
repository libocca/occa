#include <occa/internal/io.hpp>
#include <occa/internal/utils/env.hpp>
#include <occa/internal/utils/lex.hpp>
#include <occa/internal/utils/string.hpp>

#include <occa/internal/lang/file.hpp>
#include <occa/internal/lang/tokenizer.hpp>

namespace occa {
  namespace lang {
    file_t::file_t(const std::string &filename_) :
      filename(filename_),
      expandedFilename(io::expandFilename(filename_)),
      content(io::read(filename_)) {}

    file_t::file_t(const std::string &filename_,
                   const std::string &content_) :
      filename(filename_),
      expandedFilename(io::expandFilename(filename_)),
      content(content_) {}

    file_t::file_t(const bool,
                   const std::string &name) :
      filename(name),
      expandedFilename(name),
      content("") {
      dontUseRefs();
    }

    namespace originSource {
      file_t builtin(true, "(builtin)");
      file_t string(true, "(source)" );
    }

    //---[ File Origin ]----------------
    filePosition::filePosition() :
      line(1),
      lineStart(NULL),
      start(NULL),
      end(NULL) {}

    filePosition::filePosition(const char *root) :
      line(1),
      lineStart(root),
      start(root),
      end(root) {}

    filePosition::filePosition(const int line_,
                               const char *lineStart_,
                               const char *start_,
                               const char *end_) :
      line(line_),
      lineStart(lineStart_),
      start(start_),
      end(end_) {}

    filePosition::filePosition(const filePosition &other) :
      line(other.line),
      lineStart(other.lineStart),
      start(other.start),
      end(other.end) {}


    filePosition& filePosition::operator = (const filePosition &other) {
      line = other.line;
      lineStart = other.lineStart;
      start = other.start;
      end = other.end;
      return *this;
    }

    size_t filePosition::size() const {
      return (end - start);
    }

    std::string filePosition::str() const {
      if (!start) {
        return "";
      }
      return std::string(start, end - start);
    }
    //==================================

    //---[ File Origin ]----------------
    fileOrigin::fileOrigin() :
      fromInclude(true),
      file(&originSource::string),
      position(),
      up(NULL) {
      file->addRef();
    }

    fileOrigin::fileOrigin(file_t &file_) :
      fromInclude(true),
      file(&file_),
      position(file_.content.c_str()),
      up(NULL) {
      file->addRef();
    }

    fileOrigin::fileOrigin(const filePosition &position_) :
      fromInclude(true),
      file(&originSource::string),
      position(position_),
      up(NULL) {
      file->addRef();
    }

    fileOrigin::fileOrigin(file_t &file_,
                           const filePosition &position_) :
      fromInclude(true),
      file(&file_),
      position(position_),
      up(NULL) {
      file->addRef();
    }

    fileOrigin::fileOrigin(const fileOrigin &other) :
      fromInclude(other.fromInclude),
      file(other.file),
      position(other.position),
      up(other.up) {
      file->addRef();
      if (up) {
        up->addRef();
      }
    }

    fileOrigin& fileOrigin::operator = (const fileOrigin &other) {
      fromInclude = other.fromInclude;
      position    = other.position;

      setFile(*other.file);
      setUp(other.up);
      return *this;
    }

    fileOrigin::~fileOrigin() {
      clear();
    }

    void fileOrigin::clear() {
      if (file && !file->removeRef()) {
        delete file;
      }
      if (up && !up->removeRef()) {
        delete up;
      }
      file = NULL;
      up   = NULL;
    }

    bool fileOrigin::isValid() const {
      return file;
    }

    void fileOrigin::setFile(file_t &file_) {
      file_.addRef();
      if (file && !file->removeRef()) {
        delete file;
      }
      file = &file_;
    }

    void fileOrigin::setUp(fileOrigin *up_) {
      if (up_) {
        up_->addRef();
      }
      if (up && !up->removeRef()) {
        delete up;
      }
      up = up_;
    }

    void fileOrigin::push(const bool fromInclude_,
                          const fileOrigin &origin) {
      push(fromInclude_,
           *origin.file,
           origin.position);
    }

    void fileOrigin::push(const bool fromInclude_,
                          file_t &file_,
                          const filePosition &position_) {

      setUp(new fileOrigin(*this));
      up->fromInclude = fromInclude_;
      position = position_;

      setFile(file_);
    }

    void fileOrigin::pop() {
      OCCA_ERROR("Unable to call fileOrigin::pop()",
                 up != NULL);

      fromInclude = up->fromInclude;
      position    = up->position;
      setFile(*(up->file));
      setUp(up->up);
    }

    fileOrigin fileOrigin::from(const bool fromInclude_,
                                const fileOrigin &origin) {
      fileOrigin fo = origin;
      fo.push(fromInclude_, *this);
      return fo;
    }

    int fileOrigin::emptyLinesBefore(const char *pos) {
      const char *start = file->content.c_str();
      const char *end   = start + file->content.size();

      if (pos < start || end < pos) {
        return 0;
      }

      int count = 0;
      for (const char *c = pos; c >= start; --c) {
        if (!lex::isWhitespace(*c)) {
          break;
        }
        if (*c == '\n') {
          ++count;
        }
      }
      return count;
    }

    int fileOrigin::emptyLinesAfter(const char *pos) {
      const char *start = file->content.c_str();
      const char *end   = start + file->content.size();

      if (pos < start || end < pos) {
        return 0;
      }

      int count = 0;
      for (const char *c = pos; c <= end; --c) {
        if (!lex::isWhitespace(*c)) {
          break;
        }
        if (*c == '\n') {
          ++count;
        }
      }
      return count;
    }

    dim_t fileOrigin::distanceTo(const fileOrigin &origin) {
      if (file != origin.file) {
        return -1;
      }
      return (origin.position.start - position.end);
    }

    bool fileOrigin::operator == (const fileOrigin &origin) {
      if (file != origin.file) {
        return false;
      }
      return (
        position.start == origin.position.start
        && position.end == origin.position.end
      );
    }

    void fileOrigin::preprint(io::output &out) const {
      print(out, true);
    }

    void fileOrigin::postprint(io::output &out) const {
      const char *lineEnd = position.lineStart;
      lex::skipTo(lineEnd, '\n');

      const std::string line(position.lineStart,
                             lineEnd - position.lineStart);
      const std::string space(position.start - position.lineStart, ' ');

      out << line << '\n'
          << space << green("^") << '\n';
    }

    void fileOrigin::print(io::output &out,
                           const bool root) const {
      if (up) {
        up->print(out, false);
      }
      // Print file location
      out << blue(file->filename)
          ;
      if (file != &originSource::builtin) {
        out << ':' << position.line
            << ':' << (position.start - position.lineStart + 1);
      }
      out << ": ";

      if (!root) {
        if (fromInclude) {
          out << "Included file:\n";
        } else {
          out << "Expanded from macro '" << position.str() << "':\n";
        }
      }
    }

    void fileOrigin::printWarning(const std::string &message) const {
      if (env::OCCA_VERBOSE) {
        preprint(io::stderr);
        occa::printWarning(io::stderr, message);
        postprint(io::stderr);
      }
    }

    void fileOrigin::printError(const std::string &message) const {
      preprint(io::stderr);
      occa::printError(io::stderr, message);
      postprint(io::stderr);
    }
    //==================================
  }
}
