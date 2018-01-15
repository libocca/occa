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
#include "file.hpp"
#include "token.hpp"
#include "occa/tools/io.hpp"

namespace occa {
  namespace lang {
    file_t::file_t(const std::string &filename_) :
      dirname(io::dirname(filename_)),
      filename(io::basename(filename_)),
      content(io::read(filename_)) {}

    filePosition::filePosition() :
      line(1),
      lineStart(NULL),
      pos(NULL) {}

    filePosition::filePosition(const char *root) :
      line(1),
      lineStart(root),
      pos(root) {}

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
                           const filePosition &position_) :
      fromInclude(true),
      file(file_),
      position(position_),
      up(NULL) {
      if (file) {
        file->addRef();
      }
    }

    fileOrigin::fileOrigin(const fileOrigin &other) :
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

    fileOrigin& fileOrigin::operator = (const fileOrigin &other) {
      fromInclude = other.fromInclude;
      file = other.file;
      position = other.position;
      up = other.up;
      if (file) {
        file->addRef();
      }
      if (up) {
        up->addRef();
      }
      return *this;
    }

    fileOrigin::~fileOrigin() {
      if (file && !file->removeRef()) {
        delete file;
      }
      if (up && !up->removeRef()) {
        delete up;
      }
    }

    void fileOrigin::push(const bool fromInclude_,
                          file_t *file_,
                          const filePosition &position_) {
      fileOrigin &newUp = *(new fileOrigin(*this));
      newUp.fromInclude = fromInclude_;
      newUp.addRef();
      if (up) {
        up->removeRef();
      }
      up = &newUp;
      file = file_;
      position = position_;
    }

    void fileOrigin::print(printer &pout,
                           const bool root) {
      if (up) {
        up->print(pout, false);
      }
      // Print file location
      if (file) {
        pout << file->filename;
      } else {
        pout << "(source)";
      }
      pout << ':' << position.line
           << ':' << (position.pos - position.lineStart + 1)
           << ": ";
      if (!root) {
        if (fromInclude) {
          pout << "Included file:\n";
        } else {
          tokenStream stream(position.pos);
          std::string macro;
          stream.getIdentifier(macro);
          pout << "Expanded from macro '" << macro << "':\n";
        }
      }
    }
  }
}
