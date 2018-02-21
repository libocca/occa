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
#include "occa/tools/string.hpp"

#include "errorHandler.hpp"

namespace occa {
  namespace lang {
    errorHandler::errorHandler() :
      warnings(0),
      errors(0) {}

    void errorHandler::preprint(std::ostream &out) {}
    void errorHandler::postprint(std::ostream &out) {}

    void errorHandler::printNote(std::ostream &out,
                                 const std::string &message) {
      preprint(out);
      out << blue("Note: ")
          << message << '\n';
      postprint(out);
    }

    void errorHandler::printWarning(std::ostream &out,
                                    const std::string &message) {
      ++warnings;
      preprint(out);
      out << yellow("Warning: ")
          << message << '\n';
      postprint(out);
    }

    void errorHandler::printError(std::ostream &out,
                                  const std::string &message) {
      ++errors;
      preprint(out);
      out << red("Error: ")
          << message << '\n';
      postprint(out);
    }
  }
}
